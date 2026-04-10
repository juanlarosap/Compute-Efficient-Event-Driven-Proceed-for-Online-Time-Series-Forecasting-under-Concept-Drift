import torch
import torch.nn as nn
import transformers
from adapter.module.generator import AdaptGenerator
from adapter.module import down_up
import time
import torch.nn.functional as F


def normalize(W, max_norm=1):
    W_norm = torch.norm(W, dim=-1, keepdim=True)
    scale = torch.clip(max_norm / W_norm, max=1)
    return W * scale


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Proceed(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        if args.freeze:
            backbone.requires_grad_(False)
        self.backbone = add_adapters_(backbone, args)
        self.more_bias = not args.freeze
        self.generator = AdaptGenerator(backbone, args.concept_dim,
                                        activation=nn.Sigmoid if args.act == 'sigmoid' else nn.Identity,
                                        adaptive_dim=False, need_bias=self.more_bias,
                                        shared=not args.individual_generator,
                                        mid_dim=args.bottleneck_dim)
        # print(self.adapters)
        self.register_buffer('recent_batch', torch.zeros(1, args.seq_len + args.pred_len, args.enc_in), persistent=False)
        if args.ema > 0:
            self.register_buffer('recent_concept', None, persistent=True)
        self.mlp1 = nn.Sequential(Transpose(-1, -2), nn.Linear(args.seq_len, args.concept_dim), nn.GELU(),
                                  nn.Linear(args.concept_dim, args.concept_dim))
        self.mlp2 = nn.Sequential(Transpose(-1, -2), nn.Linear(args.seq_len + args.pred_len, args.concept_dim), nn.GELU(),
                                  nn.Linear(args.concept_dim, args.concept_dim))
        self.ema = args.ema
        self.flag_online_learning = False
        self.flag_update = False
        self.flag_current = False
        self.flag_basic = False

        # --- Embedding retrieval (memory bank) ---
        self.use_retrieval = getattr(args, 'use_retrieval', False)
        self.bank_size = int(getattr(args, 'bank_size', 2048))
        self.k = int(getattr(args, 'k', 8))
        self.tau = float(getattr(args, 'tau', 0.07))
        self.retrieval_alpha = float(getattr(args, 'retrieval_alpha', 0.8))

        if self.use_retrieval:
            # bank stores normalized embeddings of shape [concept_dim]
            self.register_buffer('mem_bank', torch.zeros(self.bank_size, args.concept_dim), persistent=False)
            self.mem_ptr = 0
            self.mem_count = 0

            # overhead stats
            self.retrieval_time_total = 0.0
            self.retrieval_calls = 0

    def reset_retrieval_stats(self):
        if getattr(self, 'use_retrieval', False):
            self.retrieval_time_total = 0.0
            self.retrieval_calls = 0

    @torch.no_grad()
    def _mem_add(self, emb_1d: torch.Tensor):
        """Store one normalized embedding [D] into circular buffer."""
        if not self.use_retrieval:
            return
        emb = emb_1d.detach()
        emb = F.normalize(emb, p=2, dim=-1)
        self.mem_bank[self.mem_ptr].copy_(emb)
        self.mem_ptr = (self.mem_ptr + 1) % self.bank_size
        self.mem_count = min(self.mem_count + 1, self.bank_size)

    @torch.no_grad()
    def _mem_retrieve(self, query_1d: torch.Tensor):
        """
        query_1d: normalized [D]
        returns: retrieved_ref [D] (normalized), top_sims [k]
        """
        n = int(self.mem_count)
        if n <= 0:
            return None, None
        bank = self.mem_bank[:n]  # [n, D]
        sims = torch.matmul(bank, query_1d)  # [n]
        kk = min(int(self.k), sims.numel())
        top_sims, top_idx = torch.topk(sims, k=kk, largest=True, sorted=True)  # [kk]
        w = torch.softmax(top_sims / max(float(self.tau), 1e-8), dim=0)  # [kk]
        retrieved = (bank[top_idx] * w.unsqueeze(-1)).sum(dim=0)  # [D]
        retrieved = F.normalize(retrieved, p=2, dim=-1)
        return retrieved, top_sims

    def generate_adaptation(self, x):
        # concept per batch: [B, D]
        concept = self.mlp1(x).mean(-2)

        # recent concept baseline: [D]
        recent_concept = self.mlp2(self.recent_batch).mean(-2).mean(list(range(0, self.recent_batch.dim() - 2)))

        # EMA (existing behavior)
        if self.ema > 0:
            if self.recent_concept is not None:
                recent_concept = self.recent_concept * self.ema + recent_concept * (1 - self.ema)
            if self.flag_update or self.flag_online_learning and not self.flag_current:
                self.recent_concept = recent_concept.detach()

        # Retrieval reference (smooth drift)
        ref = recent_concept  # baseline

        if self.use_retrieval and self.mem_count > 0:
            query = concept.detach().mean(dim=0)  # [D]
            query = F.normalize(query, p=2, dim=-1)

            t0 = time.perf_counter()
            retrieved, top_sims = self._mem_retrieve(query)
            t1 = time.perf_counter()
            self.retrieval_time_total += (t1 - t0)
            self.retrieval_calls += 1

            if retrieved is not None:
                # ---- MIXING ----
                alpha = float(self.retrieval_alpha)

                # scale retrieved to match magnitude of recent_concept (important!)
                scale = recent_concept.detach().norm(p=2).clamp(min=1e-6)
                retrieved_scaled = retrieved * scale

                ref = alpha * recent_concept + (1.0 - alpha) * retrieved_scaled

            # add after retrieval to avoid retrieving itself
            self._mem_add(query)

        elif self.use_retrieval:
            # bank empty: start filling
            query = concept.detach().mean(dim=0)
            query = F.normalize(query, p=2, dim=-1)
            self._mem_add(query)

        drift = concept - ref
        res = self.generator(drift, need_clip=not self.args.wo_clip)
        return res

    def forward(self, *x):
        if self.flag_basic:
            adaptations = {}
            for i, (k, adapter) in enumerate(self.generator.bottlenecks.items()):
                adaptations[k] = adapter.biases[-1] if adapter.need_bias else [None] * len(self.generator.dim_name_dict[k])
        else:
            adaptations = self.generate_adaptation(x[0])
        for out_dim, adaptation in adaptations.items():
            for i in range(len(adaptation)):
                name = self.generator.dim_name_dict[out_dim][i]
                self.backbone.get_submodule(name).assign_adaptation(adaptation[i])
        if self.args.do_predict:
            return self.backbone(*x)
        else:
            return self.backbone(*x)

    def freeze_adapter(self, freeze=True):
        for module_name in ['mlp1', 'mlp2']:
            if hasattr(self, module_name):
                getattr(self, module_name).requires_grad_(not freeze)
                getattr(self, module_name).zero_grad(set_to_none=True)
        for adapter in self.generator.bottlenecks.values():
            adapter.weights.requires_grad_(not freeze)
            adapter.weights.zero_grad(set_to_none=True)
            adapter.biases[:len(adapter.weights) - 1].requires_grad_(not freeze)
            adapter.biases[:len(adapter.weights) - 1].zero_grad(set_to_none=True)

    def freeze_bias(self, freeze=True):
        if self.more_bias:
            for adapter in self.generator.bottlenecks.values():
                adapter.biases[-1].requires_grad_(not freeze)
                adapter.biases[-1:].zero_grad(set_to_none=True)


def add_adapters_(parent_module: nn.Module, args, top_level=True):
    for name, module in parent_module.named_children():
        if args.tune_mode == 'all_down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D,
                                                                     nn.LayerNorm, nn.BatchNorm1d)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif args.tune_mode == 'down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        else:
            add_adapters_(module, args, False)
    return parent_module
