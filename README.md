# Compute-Efficient Event-Driven Proceed for Online Time Series Forecasting under Concept Drift

This repository accompanies the paper **"Compute-Efficient Event-Driven Proceed for Online Time Series Forecasting under Concept Drift"**.

The project studies lightweight, compute-efficient extensions of **PROCEED** for online time series forecasting under concept drift. In particular, it evaluates event-driven and retrieval-based modifications built on top of the original method using **iTransformer** as the forecasting backbone.

## Overview

We consider the following variants:

- **PROCEED**: original method
- **Proceed-ET** (`--use_err_gate`): error-triggered adaptation
- **Proceed-ER** (`--use_retrieval`): embedding retrieval
- **Proceed-ET+ER** (`--use_err_gate --use_retrieval`): combined error-triggered adaptation and embedding retrieval

Experiments are conducted on **ETTh2** and **Weather**, and are evaluated using:

- **MSE**
- **MAE**
- **Total runtime**
- **Milliseconds per step**
- **% updates**
- **% retrieval**

The main finding of this work is that **Proceed-ET** substantially improves computational efficiency, reducing runtime and latency while maintaining competitive predictive performance in relevant scenarios.

## Main Contributions

- A compute-efficient event-driven extension of PROCEED for online forecasting under concept drift
- An embedding retrieval variant for selective memory-based adaptation
- A combined ET+ER variant
- An experimental analysis covering both predictive accuracy and computational efficiency

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── run.py
├── settings.py
├── scripts/
├── models/
├── layers/
├── adapter/
├── data_provider/
├── dataset/
├── results/
└── [ADD OTHER RELEVANT FILES/FOLDERS HERE]
```

## Installation

Clone the repository and install the dependencies in a virtual environment:

```bash
git clone [ADD REPOSITORY URL HERE]
cd [ADD REPOSITORY NAME HERE]

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

This work uses the following datasets:

- **ETTh2**
- **Weather**

Please place the dataset files in the locations expected by `settings.py`.

Dataset preparation details:
- [ADD DATASET DOWNLOAD INSTRUCTIONS HERE]
- [ADD EXPECTED DIRECTORY STRUCTURE HERE]

## Reproducing Experiments

The commands below are **example runs** used to evaluate the different variants on **ETTh2** with **iTransformer**.

### A. PROCEED original

```bash
./.venv/bin/python -u run.py \
  --model iTransformer \
  --dataset ETTh2 --features M \
  --seq_len 96 --pred_len 96 \
  --itr 3 \
  --online_method Proceed \
  --pretrain --only_test \
  --online_learning_rate 0.0001
```

### B. Proceed-ET (error-triggered adaptation)

```bash
./.venv/bin/python -u run.py \
  --model iTransformer \
  --dataset ETTh2 --features M \
  --seq_len 96 --pred_len 96 \
  --itr 3 \
  --online_method Proceed \
  --pretrain --only_test \
  --online_learning_rate 0.0001 \
  --use_err_gate \
  --adapt_top_p 0.1332 \
  --gate_window 256 \
  --warmup_steps 200
```

### C. Proceed-ER (embedding retrieval)

```bash
./.venv/bin/python -u run.py \
  --model iTransformer \
  --dataset ETTh2 --features M \
  --seq_len 96 --pred_len 96 \
  --itr 3 \
  --online_method Proceed \
  --pretrain --only_test \
  --online_learning_rate 0.0001 \
  --use_retrieval \
  --bank_size 2048 \
  --k 8 \
  --tau 0.132209 \
  --retrieval_alpha 0.8191
```

### D. Proceed-ET+ER

```bash
./.venv/bin/python -u run.py \
  --model iTransformer \
  --dataset ETTh2 --features M \
  --seq_len 96 --pred_len 96 \
  --itr 3 \
  --online_method Proceed \
  --pretrain --only_test \
  --online_learning_rate 0.0001 \
  --use_err_gate \
  --adapt_top_p 0.1332 \
  --gate_window 256 \
  --warmup_steps 200 \
  --use_retrieval \
  --bank_size 2048 \
  --k 8 \
  --tau 0.132209 \
  --retrieval_alpha 0.8191
```

### Notes

- These commands are provided as **reference examples** for reproducing representative experiments.
- Please ensure that dataset paths are correctly configured in `settings.py`.
- Additional scripts, configurations, or dataset-specific commands can be added in `scripts/` as needed.

## Main Results

The paper reports results in terms of both forecasting accuracy and computational efficiency.

Key message:
- **Proceed-ET** consistently reduces runtime and step-level latency
- It maintains competitive accuracy in relevant online forecasting settings under concept drift


## Relation to the Original PROCEED Codebase

This repository builds upon the original **OnlineTSF / PROCEED** codebase released by the original authors.

The original README describes `OnlineTSF` as an online time series forecasting framework and states that it contains the official code of **PROCEED**. It also attributes the original PROCEED paper to **Lifan Zhao** and **Yanyan Shen**.

If you use the original implementation, please also refer to the original repository and cite the original PROCEED paper by its authors which can be found here: https://dl.acm.org/doi/10.1145/3690624.3709210


## Citation

If you use this repository, please cite:

```bibtex
[ADD BIBTEX FOR THIS PAPER HERE]
```

Please also cite the original PROCEED paper when appropriate:

```bibtex
@InProceedings{Proceed,
  author       = {Lifan Zhao and Yanyan Shen},
  booktitle    = {Proceedings of the 31st {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  title        = {Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting},
  year         = {2025},
  month        = {feb},
  publisher    = {{ACM}},
  doi          = {10.1145/3690624.3709210},
}
```# Compute-Efficient-Event-Driven-Proceed-for-Online-Time-Series-Forecasting-under-Concept-Drift
# Compute-Efficient-Event-Driven-Proceed-for-Online-Time-Series-Forecasting-under-Concept-Drift
# Compute-Efficient-Event-Driven-Proceed-for-Online-Time-Series-Forecasting-under-Concept-Drift
# Compute-Efficient-Event-Driven-Proceed-for-Online-Time-Series-Forecasting-under-Concept-Drift
# Compute-Efficient-Event-Driven-Proceed-for-Online-Time-Series-Forecasting-under-Concept-Drift
# Compute-Efficient-Event-Driven-Proceed-for-Online-Time-Series-Forecasting-under-Concept-Drift
