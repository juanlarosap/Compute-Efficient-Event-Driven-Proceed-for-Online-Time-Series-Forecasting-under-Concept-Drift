import copy
import numpy as np
import torch
import time
from tqdm import tqdm
from collections import deque

from adapter import proceed
from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp import Exp_Online


class Exp_Proceed(Exp_Online):
    def __init__(self, args):
        args = copy.deepcopy(args)
        args.merge_weights = 1
        super(Exp_Proceed, self).__init__(args)
        self.online_phases = ['val', 'test', 'online']
        self.mean_dim = args.seq_len

        # --- Cost meters (Proceed baseline) ---
        self._online_steps = 0
        self._online_updates = 0
        self._online_time = 0.0
        self._online_total_time = 0.0
        # --- Error-gate (dynamic threshold / top-p) ---
        self._gate_losses = deque(maxlen=getattr(args, "gate_window", 256))


        # print frequency (simple constant; you can later convert to an arg)
        self._log_every = 100

    def _reset_cost_meters(self):
        self._online_steps = 0
        self._online_updates = 0
        self._online_time = 0.0
        self._online_total_time = 0.0
        if hasattr(self, "_gate_losses"):
            self._gate_losses.clear()

    def _maybe_cuda_sync(self):
        """Uncomment torch.cuda.synchronize() for more accurate GPU wall-clock timing."""
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        return

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        if hasattr(self._model, 'reset_retrieval_stats'):
            self._model.reset_retrieval_stats()

        if phase == 'val' and self.args.val_online_lr:
            lr = self.model_optim.param_groups[0]['lr']
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = self.args.online_learning_rate
        self.model_optim.zero_grad()
        self._model.freeze_adapter(True)

        # --- cost timing start ---
        self._reset_cost_meters()
        self._maybe_cuda_sync()
        t0 = time.perf_counter()

        ret = super().online(online_data, target_variate, phase, show_progress)

        self._maybe_cuda_sync()
        t1 = time.perf_counter()
        self._online_total_time = (t1 - t0)
        # --- cost timing end ---

        self._model.freeze_adapter(False)
        if phase == 'val' and self.args.val_online_lr:
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = lr

        # --- cost summary ---
        steps = max(self._online_steps, 1)
        upd = self._online_updates
        upd_pct = 100.0 * upd / steps
        sec_per_step = (self._online_time / steps) if self._online_time > 0 else (self._online_total_time / steps)
        print(f"[Proceed][{phase}][DONE] steps={steps} updates={upd} updates%={upd_pct:.2f} "
              f"total_time={self._online_total_time:.2f}s sec/step={sec_per_step:.4f}")

        if hasattr(self._model, 'use_retrieval') and self._model.use_retrieval and getattr(self._model,
                                                                                           'retrieval_calls', 0) > 0:
            retr_total = float(self._model.retrieval_time_total)
            calls = int(self._model.retrieval_calls)
            ms_per = 1000.0 * retr_total / max(calls, 1)
            pct = 100.0 * retr_total / max(self._online_total_time, 1e-9)
            print(
                f"[Proceed][{phase}][Retrieval] calls={calls} total={retr_total:.3f}s ms/call={ms_per:.3f} pct_total={pct:.2f}%")

        return ret

    def update_valid(self, valid_data=None, valid_dataloader=None):
        self.phase = 'online'
        self._reset_cost_meters()
        self._maybe_cuda_sync()
        t0 = time.perf_counter()

        if valid_data is None:
            valid_data = get_dataset(self.args, 'val', self.device,
                                     wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                     **self.wrap_data_kwargs, take_post=self.args.pred_len - 1)
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.model.train()
        predictions = []
        if not self.args.joint_update_valid:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                self._model.freeze_bias(False)
                self._model.freeze_adapter(True)
                self._update_online(recent_batch, criterion, model_optim, scaler, flag_current=False)
                self._model.freeze_bias(True)
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(False)
                self._model.freeze_adapter(False)
                _, outputs = self._update_online(current_batch, criterion, model_optim, scaler, flag_current=True)
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(True)
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
            self._model.freeze_bias(False)
        else:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                self._update_online(recent_batch, criterion, model_optim, scaler, flag_current=True)
                if self.args.do_predict:
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.forward(current_batch)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                    self.model.train()
        self.model_optim.zero_grad()
        self._model.freeze_adapter(True)
        trainable_params = sum([param.nelement() if param.requires_grad else 0 for param in self._model.parameters()])
        print(f'Trainable Params: {trainable_params}', '({:.1f}%)'.format(trainable_params / self.model_params * 100))

        # --- cost summary (valid update loop) ---
        self._maybe_cuda_sync()
        t1 = time.perf_counter()
        self._online_total_time = (t1 - t0)
        steps = max(self._online_steps, 1)
        upd = self._online_updates
        upd_pct = 100.0 * upd / steps
        sec_per_step = (self._online_time / steps) if self._online_time > 0 else (self._online_total_time / steps)
        print(f"[Proceed][update_valid][DONE] steps={steps} updates={upd} updates%={upd_pct:.2f} "
              f"total_time={self._online_total_time:.2f}s sec/step={sec_per_step:.4f}")

        return predictions

    def _build_model(self, model=None, framework_class=None):
        model = super()._build_model(model, framework_class=proceed.Proceed)
        print(model)
        return model

    def _update(self, batch, criterion, optimizer, scaler=None):
        self._model.flag_update = True
        loss, outputs = super()._update(batch, criterion, optimizer, scaler)
        self._model.recent_batch = torch.cat([batch[0], batch[1]], -2)
        self._model.flag_update = False
        return loss, outputs

    def _err_gate_should_update(self, loss_val: float) -> bool:
        """Decide si hacemos optimizer.step() en este step.
        Política: actualiza solo el top-p% de losses (peores) dentro de una ventana reciente.
        Umbral dinámico = cuantíl (1 - p) del historial de losses."""
        if not getattr(self.args, "use_err_gate", False):
            return True

        p = float(getattr(self.args, "adapt_top_p", 1.0))
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False

        warm = int(getattr(self.args, "warmup_steps", 0))
        if self._online_steps <= warm:
            return True

        # Si no hay suficiente historial, actualiza para estabilizar
        if self._gate_losses is None or len(self._gate_losses) < max(10, int(0.2 * self._gate_losses.maxlen)):
            return True

        # Umbral dinámico (usa historial reciente, NO incluye el loss actual todavía)
        hist = np.asarray(self._gate_losses, dtype=np.float32)
        thr = float(np.quantile(hist, 1.0 - p))

        return loss_val >= thr

    def _update_online(self, batch, criterion, optimizer, scaler=None, flag_current=False):
        self._model.flag_online_learning = True
        self._model.flag_current = flag_current

        # timing start
        self._maybe_cuda_sync()
        step_t0 = time.perf_counter()

        # Count steps (batches processed)
        self._online_steps += 1

        # -------------------------
        # PASS 1 (barato): estimar loss SIN grad para gate
        # -------------------------
        self.model.eval()
        with torch.no_grad():
            if self.args.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs_est = self.forward(batch)
            else:
                outputs_est = self.forward(batch)

            if isinstance(outputs_est, (tuple, list)):
                outputs_est = outputs_est[0]

            # NOTA: esto asume target = batch[1], que es lo típico en tu pipeline
            y_true = batch[1]
            loss_est = criterion(outputs_est, y_true)

        loss_val = float(loss_est.detach().cpu().item())

        # Decide si actualizar (usa historial previo)
        do_update = self._err_gate_should_update(loss_val)

        # Ahora sí agrega este loss al historial (para futuras decisiones)
        self._gate_losses.append(loss_val)

        # -------------------------
        # Wrap optimizer.step para contar updates REALES
        # -------------------------
        orig_step = optimizer.step

        def counted_step(*args, **kwargs):
            self._online_updates += 1
            return orig_step(*args, **kwargs)

        optimizer.step = counted_step

        try:
            if do_update:
                # PASS 2 (con grad): usa el update "real" del padre (consistente con tu pipeline)
                self.model.train()
                loss, outputs = super()._update_online(batch, criterion, optimizer, scaler)
            else:
                # No update: devolvemos lo estimado (sin backward/step)
                self.model.train()
                loss, outputs = loss_est, outputs_est
        finally:
            optimizer.step = orig_step

        # timing end
        self._maybe_cuda_sync()
        step_t1 = time.perf_counter()
        self._online_time += (step_t1 - step_t0)

        if (self._online_steps == 1) or (self._online_steps % self._log_every == 0):
            upd_pct = 100.0 * self._online_updates / max(self._online_steps, 1)
            print(f"[Proceed][online][step {self._online_steps}] loss={loss_val:.6f} "
                  f"do_update={int(do_update)} updates%={upd_pct:.2f} sec/step={(step_t1 - step_t0):.4f}")

        self._model.recent_batch = torch.cat([batch[0], batch[1]], -2)
        self._model.flag_online_learning = False
        self._model.flag_current = not flag_current
        return loss, outputs


    def analysis_online(self):
        self._model.freeze_adapter(True)
        return super().analysis_online()

    def predict(self, path, setting, load=False):
        self.update_valid()
        res = self.online()
        np.save('./results/' + setting + '_pred.npy', np.vstack(res[-1]))
        return res