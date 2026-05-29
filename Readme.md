# Geometry, Representation, and Generalization

> **How input encoding shapes structure learning in neural networks**

Experiments for the paper *"Geometry, Representation, and Generalization: How Input Encoding Shapes Structure Learning in Neural Networks"*.

We train identical MLPs on arithmetic tasks (addition and subtraction) while varying **only** the input encoding, then measure generalization, hidden-space geometry, and representation smoothness.

---

## Key Results

| Encoding | Subtraction Gen. | Addition Gen. |
|----------|-----------------|---------------|
| Scalar   | **75.52% ± 9.34%** | **67.70% ± 9.50%** |
| Modular  | 26.39% ± 8.25% | 26.67% ± 7.31% |
| Binary   | 12.68% ± 5.55% | 10.00% ± 5.62% |
| One-Hot  | 0.33%  ± 0.78% | 0.27%  ± 0.61% |
| Random   | 0.11%  ± 0.41% | 0.33%  ± 0.66% |

---

## Repository Structure

```
geometry-generalization-experiments/
├── model.py                  # All experiments, models, plots
├── requirements.txt          # Python dependencies
├── plots/
│   ├── subtraction/          # Generated plots (subtraction)
│   └── addition/             # Generated plots (addition)
├── Subtraction_Encoders/     # Created at runtime — one folder per encoder
│   ├── scalar/
│   ├── binary/
│   └── ...
├──clear_folders.py           # cleaning the directories after multiple run
├── Addition_Encoders/        # Created at runtime — one folder per encoder
└── results/                  # Optional saved outputs
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/geometry-generalization-experiments.git
cd geometry-generalization-experiments

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run all experiments (reproducible)

```bash
python model.py
```

Runs both tasks across all 5 encoders, 30 generalisation trials each, at seed 42.
Runtime: roughly **2–4 hours on CPU**, much faster on GPU.

### 3. Common options

```bash
# Single task only
python model.py --tasks subtraction

# Custom seed
python model.py --seed 0

# Faster smoke-test
python model.py --epochs-train 100 --epochs-gen 150 --trials 5

# Reproduce paper numbers exactly
python model.py --seed 42 --trials 30 --epochs-train 300 --epochs-gen 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `42` | Global random seed |
| `--trials` | `30` | Generalisation trials per encoder |
| `--epochs-train` | `300` | Epochs for Experiments 1 & 2 |
| `--epochs-gen` | `500` | Epochs per generalisation trial |
| `--tasks` | `subtraction addition` | Tasks to run |

---

## What Each Run Produces

For every encoder × task combination, the script runs and saves:

| Experiment | Output |
|-----------|--------|
| **Exp 1** — Train on ℕ only | Printed accuracy + boundary analysis |
| **Exp 2** — Train on ℤ (full domain) | Printed accuracy |
| **Training curves** | `<task>_experiment_run<id>.png` — 4-panel: loss, accuracy, scatter of unseen Z predictions, confidence histogram |
| **Hidden space PCA** | `hidden_space_<task>_run<id>.png` — model_N vs model_Z, colored by ℕ/ℤ domain |
| **Arc hypothesis** | `arc_hypothesis_<task>_run<id>.png` — PCA colored by result value |
| **Threshold experiment** | `threshold_<task>_run<id>.png` — generalization vs % of data used for training (10%→90%) |
| **Exp 3** — 30-trial generalisation | Printed mean ± std |
| **Correlation analysis** | Printed — PCA dims vs result, a, b, a+b, a×b, \|result\| |
| **Outside-range test** | Printed — predictions on a,b values never seen during training |
| **Smoothness metric** | Printed — Pearson r between input-space and hidden-space distances |

Plus one **final comparison bar chart** per task across all encoders.

---

## Encodings

| Encoding | Representation | Geometry preserved |
|----------|---------------|-------------------|
| `scalar` | Linear map to `[-1, 1]` | Full numerical proximity |
| `modular` | Sin/cos of cyclic angle | Local continuity, cyclic topology |
| `binary` | 5-bit binary vector | Partial compositional structure |
| `one_hot` | Orthogonal basis vector | None |
| `random` | Fixed random 16-d vector (seeded) | None |

---

## Model Architecture

All experiments use the **same** architecture — only the encoder changes:

```
Input: [encode(a) ‖ encode(b)]
  └─ Linear(input_dim×2 → 64) → ReLU
  └─ Linear(64 → 64)          → ReLU
  └─ Linear(64 → 21)          # 21 classes: results in [-10, 10]

Loss:      CrossEntropyLoss
Optimizer: Adam (lr=0.01)
Batch:     32
```

---

## Reproducibility

All randomness is controlled via `--seed`, which seeds Python `random`, NumPy, and PyTorch. The `random` encoder's fixed vectors are always generated from seed `0` regardless of `--seed`, so the encoding is identical across runs.

To reproduce the paper numbers exactly:

```bash
python model.py --seed 42 --trials 30 --epochs-train 300 --epochs-gen 500
```

---

## Citation

```bibtex
@misc{geometry-generalization,
  title  = {Geometry, Representation, and Generalization:
             How Input Encoding Shapes Structure Learning in Neural Networks},
  author = {[Author]},
  year   = {2025},
  url    = {https://github.com/<your-username>/geometry-generalization-experiments}
}
```

---

## License

MIT