import os
import re
import csv
import json
import time
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import optuna


# --------- Configuración ---------
RUN_PY = "run.py"
PYTHON = "./.venv/bin/python"  # usa tu venv del proyecto

STUDY_NAME_PREFIX = "proceed_combo_ETTm1"
STUDY_NAME_STAGE1 = f"{STUDY_NAME_PREFIX}_stage1"
STUDY_NAME_STAGE2 = f"{STUDY_NAME_PREFIX}_stage2"

CSV_PATH_STAGE1 = "optuna_trials_combo_ETTm1_stage1.csv"
CSV_PATH_STAGE2 = "optuna_trials_combo_ETTm1_stage2.csv"

LOG_DIR_STAGE1 = "optuna_logs_combo_ETTm1_stage1"
LOG_DIR_STAGE2 = "optuna_logs_combo_ETTm1_stage2"

STORAGE_STAGE1 = f"sqlite:///{STUDY_NAME_STAGE1}.db"
STORAGE_STAGE2 = f"sqlite:///{STUDY_NAME_STAGE2}.db"

# Baselines / constraints
MSE_BASE = 0.865836042045331
MSE_MAX = MSE_BASE * 1.01  # +1%
TIME_BASE = 370.57

# Bounds default (amplios). Stage 2 se acota automáticamente usando Stage 1
DEFAULT_BOUNDS = {
    "retrieval_alpha": (0.55, 1.0),
    "tau": (1e-3, 0.2),          # log
    "adapt_top_p": (0.05, 0.99),
}

# Comando base: ajusta aquí si cambias dataset/model/flags
BASE_ARGS = [
    "-u", RUN_PY,
    "--model", "iTransformer",
    "--dataset", "ETTm1",
    "--features", "M",
    "--seq_len", "96",
    "--pred_len", "96",
    "--itr", "1",
    "--online_method", "Proceed",
    "--pretrain",
    "--only_test",
    "--online_learning_rate", "0.0001",
    "--use_retrieval",
    "--bank_size", "2048",
    "--k", "8",
    "--use_err_gate",
    "--gate_window", "256",
    "--warmup_steps", "200",
]

# Regex para parsear métricas
RE_MSE_MAE = re.compile(r"mse:\s*([0-9.eE+-]+)\s*,\s*mae:\s*([0-9.eE+-]+)")
RE_DONE = re.compile(
    r"\[Proceed\]\[test\]\[DONE\].*total_time=([0-9.eE+-]+)s\s+sec/step=([0-9.eE+-]+)"
)

# (Opcional) si quieres incluir retrieval ms/call, lo parseamos también
RE_RETR = re.compile(
    r"\[Proceed\]\[test\]\[Retrieval\].*ms/call=([0-9.eE+-]+)"
)


@dataclass
class TrialResult:
    mse: float
    mae: float
    total_time_s: float
    sec_per_step: float
    ms_per_call: Optional[float]
    raw_output: str


def ensure_outputs():
    os.makedirs(LOG_DIR_STAGE1, exist_ok=True)
    os.makedirs(LOG_DIR_STAGE2, exist_ok=True)

    if not os.path.exists(CSV_PATH_STAGE1):
        with open(CSV_PATH_STAGE1, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial_number",
                "alpha",
                "tau",
                "adapt_top_p",
                "mse",
                "mae",
                "total_time_s",
                "sec_per_step",
                "ms_per_call",
                "return_code",
                "cmd"
            ])

    if not os.path.exists(CSV_PATH_STAGE2):
        with open(CSV_PATH_STAGE2, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial_number",
                "alpha",
                "tau",
                "adapt_top_p",
                "mse",
                "mae",
                "total_time_s",
                "sec_per_step",
                "ms_per_call",
                "return_code",
                "cmd"
            ])


def run_once(alpha: float, tau: float, adapt_top_p: float, timeout_s: int = 60 * 60) -> Tuple[int, TrialResult, str]:
    """
    Ejecuta run.py con alpha/tau/adapt_top_p. Devuelve (return_code, TrialResult, cmd_str).
    """
    cmd = [PYTHON] + BASE_ARGS + [
        "--retrieval_alpha", f"{alpha}",
        "--tau", f"{tau}",
        "--use_err_gate",
        "--adapt_top_p", f"{adapt_top_p}",
    ]

    cmd_str = " ".join(shlex.quote(x) for x in cmd)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        env=dict(os.environ),
    )
    out = proc.stdout

    # Parseo robusto: buscamos la última ocurrencia
    mse_mae = None
    for m in RE_MSE_MAE.finditer(out):
        mse_mae = (float(m.group(1)), float(m.group(2)))

    done = None
    for m in RE_DONE.finditer(out):
        done = (float(m.group(1)), float(m.group(2)))

    ms_call = None
    for m in RE_RETR.finditer(out):
        ms_call = float(m.group(1))

    if mse_mae is None:
        raise RuntimeError("No pude parsear MSE/MAE del output.\n" + out[-2000:])
    if done is None:
        raise RuntimeError("No pude parsear total_time/sec_step del output.\n" + out[-2000:])

    res = TrialResult(
        mse=mse_mae[0],
        mae=mse_mae[1],
        total_time_s=done[0],
        sec_per_step=done[1],
        ms_per_call=ms_call,
        raw_output=out
    )
    return proc.returncode, res, cmd_str


def append_csv(csv_path: str, trial_number: int, alpha: float, tau: float, adapt_top_p: float,
               res: TrialResult, return_code: int, cmd_str: str):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trial_number,
            alpha,
            tau,
            adapt_top_p,
            res.mse,
            res.mae,
            res.total_time_s,
            res.sec_per_step,
            res.ms_per_call if res.ms_per_call is not None else "",
            return_code,
            cmd_str
        ])


def save_log(log_dir: str, trial_number: int, alpha: float, tau: float, adapt_top_p: float, res: TrialResult):
    path = os.path.join(log_dir, f"trial_{trial_number:03d}_a{alpha}_t{tau}_p{adapt_top_p}.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write(res.raw_output)


def set_common_user_attrs(trial: optuna.Trial, return_code: int, cmd_str: str, wall_clock_s: float,
                          res: Optional[TrialResult] = None):
    trial.set_user_attr("wall_clock_s", float(wall_clock_s))
    trial.set_user_attr("cmd", cmd_str)
    trial.set_user_attr("return_code", int(return_code))

    if res is None:
        return

    trial.set_user_attr("mae", float(res.mae))
    trial.set_user_attr("mse", float(res.mse))
    trial.set_user_attr("total_time_s", float(res.total_time_s))
    trial.set_user_attr("sec_per_step", float(res.sec_per_step))
    trial.set_user_attr("ms_per_call", float(res.ms_per_call) if res.ms_per_call is not None else None)
    trial.set_user_attr("speedup_vs_base", float(TIME_BASE / max(float(res.total_time_s), 1e-12)))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _quantile_sorted(xs: List[float], q: float) -> float:
    if not xs:
        raise ValueError("Empty list for quantile.")
    idx = int(round((len(xs) - 1) * q))
    idx = int(_clamp(idx, 0, len(xs) - 1))
    return xs[idx]


def compute_bounds_from_trials(
    trials: List[optuna.trial.FrozenTrial],
    default_bounds: Dict[str, Tuple[float, float]],
    q_low: float = 0.10,
    q_high: float = 0.90,
    pad_frac: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """
    Bounds = quantiles (q_low..q_high) of Stage 1 params + small padding,
    clamped to default_bounds. If insufficient trials, fallback to default.
    """
    vals: Dict[str, List[float]] = {k: [] for k in default_bounds.keys()}
    for t in trials:
        for k in vals.keys():
            if k in t.params:
                vals[k].append(float(t.params[k]))

    out: Dict[str, Tuple[float, float]] = {}
    for k, (dlo, dhi) in default_bounds.items():
        xs = sorted(vals.get(k, []))
        if len(xs) < 5:
            out[k] = (dlo, dhi)
            continue

        lo = _quantile_sorted(xs, q_low)
        hi = _quantile_sorted(xs, q_high)
        if hi <= lo:
            out[k] = (dlo, dhi)
            continue

        pad = (hi - lo) * pad_frac
        lo2 = _clamp(lo - pad, dlo, dhi)
        hi2 = _clamp(hi + pad, dlo, dhi)

        if hi2 <= lo2:
            out[k] = (dlo, dhi)
        else:
            out[k] = (lo2, hi2)

    return out


# -------------------------
# Stage 1 objective: minimize MSE
# -------------------------
def objective_stage1(trial: optuna.Trial) -> float:
    alpha = trial.suggest_float("retrieval_alpha", DEFAULT_BOUNDS["retrieval_alpha"][0], DEFAULT_BOUNDS["retrieval_alpha"][1])
    tau = trial.suggest_float("tau", DEFAULT_BOUNDS["tau"][0], DEFAULT_BOUNDS["tau"][1], log=True)
    adapt_top_p = trial.suggest_float("adapt_top_p", DEFAULT_BOUNDS["adapt_top_p"][0], DEFAULT_BOUNDS["adapt_top_p"][1])

    t0 = time.perf_counter()
    try:
        return_code, res, cmd_str = run_once(alpha=alpha, tau=tau, adapt_top_p=adapt_top_p)
    except Exception as e:
        t1 = time.perf_counter()
        # Hard fail -> record minimal attrs
        set_common_user_attrs(trial, return_code=999, cmd_str=f"EXCEPTION: {e}", wall_clock_s=t1 - t0, res=None)
        return 1e9
    t1 = time.perf_counter()

    if return_code != 0 or res is None:
        set_common_user_attrs(trial, return_code=return_code, cmd_str=cmd_str, wall_clock_s=t1 - t0, res=None)
        return 1e9

    # Persist
    save_log(LOG_DIR_STAGE1, trial.number, alpha, tau, adapt_top_p, res)
    append_csv(CSV_PATH_STAGE1, trial.number, alpha, tau, adapt_top_p, res, return_code, cmd_str)

    set_common_user_attrs(trial, return_code=return_code, cmd_str=cmd_str, wall_clock_s=t1 - t0, res=res)

    return float(res.mse)


# -------------------------
# Stage 2 objective: minimize time with MSE constraint penalty
# -------------------------
def make_objective_stage2(bounds: Dict[str, Tuple[float, float]]):
    a_lo, a_hi = bounds["retrieval_alpha"]
    t_lo, t_hi = bounds["tau"]
    p_lo, p_hi = bounds["adapt_top_p"]

    def objective_stage2(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("retrieval_alpha", a_lo, a_hi)
        tau = trial.suggest_float("tau", t_lo, t_hi, log=True)
        adapt_top_p = trial.suggest_float("adapt_top_p", p_lo, p_hi)

        t0 = time.perf_counter()
        try:
            return_code, res, cmd_str = run_once(alpha=alpha, tau=tau, adapt_top_p=adapt_top_p)
        except Exception as e:
            t1 = time.perf_counter()
            set_common_user_attrs(trial, return_code=999, cmd_str=f"EXCEPTION: {e}", wall_clock_s=t1 - t0, res=None)
            return 1e9
        t1 = time.perf_counter()

        if return_code != 0 or res is None:
            set_common_user_attrs(trial, return_code=return_code, cmd_str=cmd_str, wall_clock_s=t1 - t0, res=None)
            return 1e9

        # Persist
        save_log(LOG_DIR_STAGE2, trial.number, alpha, tau, adapt_top_p, res)
        append_csv(CSV_PATH_STAGE2, trial.number, alpha, tau, adapt_top_p, res, return_code, cmd_str)

        set_common_user_attrs(trial, return_code=return_code, cmd_str=cmd_str, wall_clock_s=t1 - t0, res=res)

        mse = float(res.mse)
        total_time = float(res.total_time_s)

        overflow = max(0.0, mse - MSE_MAX)
        penalty = (overflow / MSE_MAX) ** 2

        return total_time * (1.0 + 50.0 * penalty)

    return objective_stage2


def main():
    ensure_outputs()

    # ---- Stage 1 ----
    n_trials_stage1 = 60
    print(f"[Optuna] Stage 1 (min MSE) | Study: {STUDY_NAME_STAGE1} | Trials: {n_trials_stage1}")

    sampler1 = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True, n_startup_trials=25)
    pruner1 = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)

    study1 = optuna.create_study(
        study_name=STUDY_NAME_STAGE1,
        storage=STORAGE_STAGE1,
        direction="minimize",
        sampler=sampler1,
        pruner=pruner1,
        load_if_exists=True,
    )
    study1.optimize(objective_stage1, n_trials=n_trials_stage1, gc_after_trial=True)

    completed1 = [t for t in study1.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible1 = [t for t in completed1 if t.user_attrs.get("mse", 1e9) <= MSE_MAX]

    # Usamos trials factibles para acotar bounds; si hay pocos, usamos top por MSE
    completed1_sorted = sorted(completed1, key=lambda t: float(t.user_attrs.get("mse", 1e9)))
    top_k_enqueue = 20
    top_trials = completed1_sorted[:max(1, top_k_enqueue)]

    ref_for_bounds = feasible1 if len(feasible1) >= 5 else top_trials
    bounds2 = compute_bounds_from_trials(ref_for_bounds, DEFAULT_BOUNDS, q_low=0.10, q_high=0.90, pad_frac=0.05)

    print("[Optuna] Stage 2 bounds:", json.dumps(bounds2, indent=2))

    # ---- Stage 2 ----
    n_trials_stage2 = 80
    print(f"[Optuna] Stage 2 (min time w/ MSE<=+1% penalty) | Study: {STUDY_NAME_STAGE2} | Trials: {n_trials_stage2}")

    sampler2 = optuna.samplers.TPESampler(seed=43, multivariate=True, group=True, n_startup_trials=25)
    pruner2 = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)

    study2 = optuna.create_study(
        study_name=STUDY_NAME_STAGE2,
        storage=STORAGE_STAGE2,
        direction="minimize",
        sampler=sampler2,
        pruner=pruner2,
        load_if_exists=True,
    )

    # Warm-start: encolamos los mejores (por MSE) de Stage 1
    for t in top_trials:
        params = t.params
        if all(k in params for k in ["retrieval_alpha", "tau", "adapt_top_p"]):
            study2.enqueue_trial({
                "retrieval_alpha": params["retrieval_alpha"],
                "tau": params["tau"],
                "adapt_top_p": params["adapt_top_p"],
            })

    objective_stage2 = make_objective_stage2(bounds2)
    study2.optimize(objective_stage2, n_trials=n_trials_stage2, gc_after_trial=True)

    # ---- Resumen final ----
    completed2 = [t for t in study2.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible2 = [t for t in completed2 if t.user_attrs.get("mse", 1e9) <= MSE_MAX]

    print("\n===== SUMMARY =====")
    print(f"MSE_BASE={MSE_BASE:.6f}  MSE_MAX(+1%)={MSE_MAX:.6f}  TIME_BASE={TIME_BASE:.2f}s")

    # Best Stage 1 MSE
    best1 = study1.best_trial
    print("\n[Stage 1] Best MSE trial")
    print("  number:", best1.number)
    print("  mse:", best1.value)
    print("  params:", best1.params)
    print("  attrs:", {k: best1.user_attrs.get(k) for k in ["mae", "total_time_s", "sec_per_step", "ms_per_call", "speedup_vs_base"]})

    if feasible2:
        best_feasible = min(feasible2, key=lambda t: float(t.user_attrs["total_time_s"]))
        bt = float(best_feasible.user_attrs["total_time_s"])
        bm = float(best_feasible.user_attrs["mse"])
        sp = TIME_BASE / bt
        red = 100.0 * (1.0 - bt / TIME_BASE)
        print("\n[Stage 2] Best FEASIBLE (mse<=MSE_MAX) by time")
        print("  number:", best_feasible.number)
        print("  time_s:", bt)
        print("  mse:", bm)
        print("  speedup:", sp)
        print("  reduction_%:", red)
        print("  params:", best_feasible.params)
    else:
        best_any = study2.best_trial
        print("\n[Stage 2] No feasible trials found under +1% MSE.")
        print("Best (penalized objective) trial:")
        print("  number:", best_any.number)
        print("  objective:", best_any.value)
        print("  params:", best_any.params)
        print("  attrs:", {k: best_any.user_attrs.get(k) for k in ["mse", "mae", "total_time_s", "sec_per_step", "ms_per_call", "speedup_vs_base"]})

    # Guardar resumen JSON (Stage 1 + Stage 2)
    summary = {
        "baseline": {"mse_base": MSE_BASE, "mse_max": MSE_MAX, "time_base_s": TIME_BASE},
        "stage1": {
            "study": STUDY_NAME_STAGE1,
            "best_trial_number": best1.number,
            "best_mse": best1.value,
            "best_params": best1.params,
            "best_attrs": best1.user_attrs,
        },
        "stage2": {
            "study": STUDY_NAME_STAGE2,
            "n_feasible": len(feasible2),
        }
    }
    if feasible2:
        summary["stage2"]["best_feasible_by_time"] = {
            "trial_number": best_feasible.number,
            "time_s": float(best_feasible.user_attrs["total_time_s"]),
            "mse": float(best_feasible.user_attrs["mse"]),
            "params": best_feasible.params,
            "attrs": best_feasible.user_attrs,
        }
    else:
        summary["stage2"]["best_penalized"] = {
            "trial_number": study2.best_trial.number,
            "objective": float(study2.best_trial.value),
            "params": study2.best_trial.params,
            "attrs": study2.best_trial.user_attrs,
        }

    with open("optuna_best_summary_two_stage.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
