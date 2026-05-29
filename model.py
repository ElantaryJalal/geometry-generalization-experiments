import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy import corrcoef
import random
import os
import time
import argparse
from scipy.stats import pearsonr

# ─────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────
# CLI ARGUMENTS
# ─────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Geometry & Generalization Experiments")
    p.add_argument('--seed',         type=int, default=42,
                   help='Global random seed (default: 42)')
    p.add_argument('--trials',       type=int, default=30,
                   help='Generalisation trials per encoder (default: 30)')
    p.add_argument('--epochs-train', type=int, default=300,
                   help='Epochs for Exp 1 & 2 (default: 300)')
    p.add_argument('--epochs-gen',   type=int, default=500,
                   help='Epochs per generalisation trial (default: 500)')
    p.add_argument('--tasks', nargs='+',
                   default=['subtraction', 'addition'],
                   choices=['subtraction', 'addition'],
                   help='Tasks to run (default: both)')
    return p.parse_args()

args = parse_args()
set_seed(args.seed)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
MIN_VAL = -10
MAX_VAL = 10
OUTPUT_SIZE = MAX_VAL - MIN_VAL + 1  # 21 possible results
HIDDEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ENCODING MODES
MODES = ['scalar', 'binary', 'random', 'modular', 'one_hot']

# Fixed random vectors — seeded so they are identical across runs
_fixed_rng = np.random.RandomState(0)
FIXED_RANDOM_VECS = {i: _fixed_rng.randn(16).astype(np.float32) for i in range(21)}

# Number of trials for robust generalisation test
N_TRIALS_GEN = args.trials

# Unique run identifier
RUN_ID = os.environ.get('RUN_ID', str(int(time.time())))

# Global storage for final comparison (mean ± std)
comparison_stats_subtraction = {}
comparison_stats_addition = {}

print(f"Device : {DEVICE}")
print(f"Seed   : {args.seed}")
print(f"Trials : {N_TRIALS_GEN}")
print(f"Tasks  : {args.tasks}")

# ─────────────────────────────────────────
# ENCODING LOGIC (unchanged from original)
# ─────────────────────────────────────────
def encode(n, mode, min_val=MIN_VAL, max_val=MAX_VAL):
    if mode == 'scalar':
        normalized = (n - min_val) / (max_val - min_val) * 2 - 1
        return [normalized]
    elif mode == 'binary':
        return [int(x) for x in format(n, '05b')]
    elif mode == 'random':
        return FIXED_RANDOM_VECS[n].tolist()
    elif mode == 'modular':
        angle = (n / (max_val + 1)) * 2 * np.pi
        return [np.sin(angle), np.cos(angle)]
    elif mode == 'one_hot':
        vec = [0.0] * 21
        if n < 21:
            vec[n] = 1.0
        return vec

def get_input_dim(mode):
    return len(encode(0, mode))

def result_to_index(result, min_val=MIN_VAL):
    return result - min_val

def index_to_result(idx, min_val=MIN_VAL):
    return idx + min_val

def prepare_tensors(problems, mode, result_min_val=MIN_VAL):
    a_vals = torch.tensor([encode(p[0], mode) for p in problems], dtype=torch.float32)
    b_vals = torch.tensor([encode(p[1], mode) for p in problems], dtype=torch.float32)
    targets = torch.tensor([result_to_index(p[2], min_val=result_min_val) for p in problems], dtype=torch.long)
    return a_vals.to(DEVICE), b_vals.to(DEVICE), targets.to(DEVICE)

# ─────────────────────────────────────────
# MODEL (identical to original)
# ─────────────────────────────────────────
class SubtractionModel(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN, output_size=OUTPUT_SIZE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size)
        )

    def forward(self, a, b):
        x = torch.cat([a, b], dim=-1)
        return self.network(x)

# ─────────────────────────────────────────
# TRAINING & PREDICTION (original signatures)
# ─────────────────────────────────────────
def train(model, problems, mode_str, epochs=500, lr=0.01, verbose=True, batch_size=32, result_min_val=MIN_VAL):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses, accuracies = [], []

    a_train, b_train, targets = prepare_tensors(problems, mode_str, result_min_val=result_min_val)
    n_samples = len(problems)

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_samples)
        total_loss, correct = 0, 0

        for i in range(0, n_samples, batch_size):
            idx = indices[i:i+batch_size]
            a_batch, b_batch, target_batch = a_train[idx], b_train[idx], targets[idx]

            output = model(a_batch, b_batch)
            loss = loss_fn(output, target_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == target_batch).sum().item()

        avg_loss = total_loss / n_samples
        accuracy = correct / n_samples
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")

    return losses, accuracies

@torch.no_grad()
def predict(model, a, b, mode_str, result_min_val=MIN_VAL):
    model.eval()
    a_vec = torch.tensor([encode(a, mode_str)], dtype=torch.float32).to(DEVICE)
    b_vec = torch.tensor([encode(b, mode_str)], dtype=torch.float32).to(DEVICE)
    output = model(a_vec, b_vec)
    return index_to_result(torch.argmax(output).item(), min_val=result_min_val)

@torch.no_grad()
def predict_with_confidence(model, a, b, mode_str, result_min_val=MIN_VAL):
    model.eval()
    a_vec = torch.tensor([encode(a, mode_str)], dtype=torch.float32).to(DEVICE)
    b_vec = torch.tensor([encode(b, mode_str)], dtype=torch.float32).to(DEVICE)
    output = torch.softmax(model(a_vec, b_vec), dim=-1)
    idx = torch.argmax(output).item()
    return index_to_result(idx, min_val=result_min_val), output.max().item()

@torch.no_grad()
def get_hidden(model, a, b, mode_str):
    model.eval()
    a_vec = torch.tensor([encode(a, mode_str)], dtype=torch.float32).to(DEVICE)
    b_vec = torch.tensor([encode(b, mode_str)], dtype=torch.float32).to(DEVICE)
    x = torch.cat([a_vec, b_vec], dim=-1)
    hidden = torch.relu(model.network[0](x))
    return hidden.cpu().numpy().flatten()

# ─────────────────────────────────────────
# DATASET GENERATION (supports subtraction & addition)
# ─────────────────────────────────────────
def make_dataset(task='subtraction', max_val=10, domain='Z'):
    problems = []
    for a in range(max_val + 1):
        for b in range(max_val + 1):
            if task == 'subtraction':
                result = a - b
            else:  # addition
                result = a + b
            if domain == 'N' and result < 0:
                continue
            problems.append((a, b, result))
    return problems

# ─────────────────────────────────────────
# SMOOTHNESS METRIC
# ─────────────────────────────────────────
def compute_smoothness(model, problems, mode, result_min_val=MIN_VAL):
    input_diffs, hidden_diffs = [], []
    pairs = random.sample(problems, min(200, len(problems)))
    for (a1,b1,_), (a2,b2,_) in zip(pairs, pairs[1:]):
        inp1 = np.array(encode(a1, mode) + encode(b1, mode))
        inp2 = np.array(encode(a2, mode) + encode(b2, mode))
        input_diff = np.linalg.norm(inp1 - inp2)
        h1 = get_hidden(model, a1, b1, mode)
        h2 = get_hidden(model, a2, b2, mode)
        hidden_diff = np.linalg.norm(h1 - h2)
        input_diffs.append(input_diff)
        hidden_diffs.append(hidden_diff)
    if len(input_diffs) < 2:
        return 0.0
    corr, _ = pearsonr(input_diffs, hidden_diffs)
    return corr

# ─────────────────────────────────────────
# EXPERIMENT RUNNER (preserves original plot styles)
# ─────────────────────────────────────────
def run_task_experiments(task_name, result_min, result_max, output_size, comparison_dict,
                         epochs_train=300, epochs_gen=500):
    print(f"\n\n{'#'*70}")
    print(f"### RUNNING EXPERIMENTS FOR TASK: {task_name.upper()}")
    print(f"{'#'*70}\n")

    for mode in MODES:
        print(f"\n\n{'='*70}")
        print(f"### ENCODER: {mode.upper()}  |  TASK: {task_name}")
        print(f"{'='*70}\n")

        # Output directory: e.g., Subtraction_Encoders/scalar/ or Addition_Encoders/scalar/
        plot_dir = f"{task_name.capitalize()}_Encoders/{mode}/"
        os.makedirs(plot_dir, exist_ok=True)

        current_input_dim = get_input_dim(mode)

        # Datasets
        N_problems = make_dataset(task=task_name, domain='N')
        Z_problems = make_dataset(task=task_name, domain='Z')
        print(f"Problems in N domain: {len(N_problems)}")
        print(f"Problems in Z domain: {len(Z_problems)}")
        print(f"New problems Z adds:  {len(Z_problems) - len(N_problems)}\n")

        # ───────── EXPERIMENT 1: Train on N only ─────────
        print("=" * 50)
        print(f"EXPERIMENT 1: Training on N domain only ({mode})")
        print("=" * 50)

        model_N = SubtractionModel(current_input_dim, output_size=output_size)
        losses_N, acc_N = train(model_N, N_problems, mode, epochs=epochs_train, result_min_val=result_min)

        print("\nTesting on N domain problems:")
        for a, b in [(3, 2), (5, 1), (4, 4), (10, 3)]:
            p = predict(model_N, a, b, mode, result_min_val=result_min)
            correct = a - b if task_name == 'subtraction' else a + b
            print(f"  {a} {'-' if task_name=='subtraction' else '+'} {b} = {p} (correct: {correct})")

        print("\nmodel_N on Z problems it was NEVER trained on:")
        for a, b in [(2, 5), (0, 7), (1, 9), (3, 8)]:
            p = predict(model_N, a, b, mode, result_min_val=result_min)
            correct = a - b if task_name == 'subtraction' else a + b
            print(f"  {a} {'-' if task_name=='subtraction' else '+'} {b} = {p} (correct: {correct}, error: {p - correct})")

        print("\nBOUNDARY ANALYSIS:")
        for a, b in [(5,5),(4,5),(3,5),(2,5),(1,5),(0,5)]:
            p, conf = predict_with_confidence(model_N, a, b, mode, result_min_val=result_min)
            correct = a - b if task_name == 'subtraction' else a + b
            print(f"  {a} {'-' if task_name=='subtraction' else '+'} {b} = {correct:3d} | model predicts: {p:3d} | confidence: {conf:.2%}")

        # ───────── EXPERIMENT 2: Train on Z ─────────
        print("\n" + "=" * 50)
        print(f"EXPERIMENT 2: Training on Z domain ({mode})")
        print("=" * 50)

        model_Z = SubtractionModel(current_input_dim, output_size=output_size)
        losses_Z, acc_Z = train(model_Z, Z_problems, mode, epochs=epochs_train, result_min_val=result_min)

        print("\nmodel_Z on N problems:")
        for a, b in [(3, 2), (5, 1), (4, 4), (10, 3)]:
            p = predict(model_Z, a, b, mode, result_min_val=result_min)
            correct = a - b if task_name == 'subtraction' else a + b
            print(f"  {a} {'-' if task_name=='subtraction' else '+'} {b} = {p} (correct: {correct})")

        print("\nmodel_Z on Z problems:")
        for a, b in [(2, 5), (0, 7), (1, 9), (3, 8)]:
            p = predict(model_Z, a, b, mode, result_min_val=result_min)
            correct = a - b if task_name == 'subtraction' else a + b
            print(f"  {a} {'-' if task_name=='subtraction' else '+'} {b} = {p} (correct: {correct})")

        # ───────── INTERNAL REPRESENTATIONS ─────────
        print("\n" + "=" * 50)
        print("INTERNAL REPRESENTATIONS")
        print("=" * 50)

        for a, b in [(5, 2), (3, 7), (0, 4)]:
            h_N = get_hidden(model_N, a, b, mode)
            h_Z = get_hidden(model_Z, a, b, mode)
            result = (a - b) if task_name == 'subtraction' else (a + b)
            domain = "ℕ" if result >= 0 else "ℤ only"
            print(f"\n{a} {'-' if task_name=='subtraction' else '+'} {b} = {result} ({domain})")
            print(f"  model_N: mean {h_N.mean():.4f}, std {h_N.std():.4f}")
            print(f"  model_Z: mean {h_Z.mean():.4f}, std {h_Z.std():.4f}")
            print(f"  distance: {np.linalg.norm(h_N - h_Z):.4f}")

        # ───────── VISUALIZE TRAINING CURVES ─────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'N to Z — {mode} encoding ({task_name})', fontsize=14)

        axes[0,0].plot(losses_N, label='model_N', color='#1D9E75')
        axes[0,0].plot(losses_Z, label='model_Z', color='#7F77DD')
        axes[0,0].set_xlabel('epoch')
        axes[0,0].set_ylabel('loss')
        axes[0,0].set_title('training loss')
        axes[0,0].legend()

        axes[0,1].plot(acc_N, label='model_N', color='#1D9E75')
        axes[0,1].plot(acc_Z, label='model_Z', color='#7F77DD')
        axes[0,1].set_xlabel('epoch')
        axes[0,1].set_ylabel('accuracy')
        axes[0,1].set_title('training accuracy')
        axes[0,1].legend()

        # Scatter for unseen negative results
        results_true, results_predicted = [], []
        for a, b, result in Z_problems:
            if result < 0:
                p = predict(model_N, a, b, mode, result_min_val=result_min)
                results_true.append(result)
                results_predicted.append(p)

        axes[1,0].scatter(results_true, results_predicted, alpha=0.5, color='#E24B4A', s=20)
        axes[1,0].plot([result_min,0], [result_min,0], 'k--', alpha=0.3, label='perfect')
        axes[1,0].set_xlabel('true result (negative)')
        axes[1,0].set_ylabel('model_N prediction')
        axes[1,0].set_title('model_N on unseen Z problems')
        axes[1,0].legend()

        def get_confidences(model, problems, m, rmin):
            return [predict_with_confidence(model, a, b, m, result_min_val=rmin)[1] for a, b, _ in problems]

        axes[1,1].hist(get_confidences(model_N, N_problems, mode, result_min), bins=20, alpha=0.6,
                       label='model_N on N', color='#1D9E75')
        axes[1,1].hist(get_confidences(model_Z, N_problems, mode, result_min), bins=20, alpha=0.6,
                       label='model_Z on N', color='#7F77DD')
        if task_name == 'subtraction':
            neg_problems = [p for p in Z_problems if p[2] < 0]
            axes[1,1].hist(get_confidences(model_Z, neg_problems, mode, result_min),
                           bins=20, alpha=0.6, label='model_Z on Z only', color='#E24B4A')
        axes[1,1].set_xlabel('confidence')
        axes[1,1].set_ylabel('count')
        axes[1,1].set_title('model confidence by domain')
        axes[1,1].legend()

        plt.tight_layout()
        fname = f'{plot_dir}{task_name}_experiment_run{RUN_ID}.png'
        plt.savefig(fname, dpi=150)
        plt.clf()
        plt.close()
        print(f"\nPlot saved: {fname}")

        # ───────── EXPERIMENT 3: Generalisation with MULTIPLE TRIALS ─────────
        print("\n" + "=" * 50)
        print(f"EXPERIMENT 3: Generalisation test with {N_TRIALS_GEN} trials ({mode})")
        print("=" * 50)

        gen_accuracies = []
        for trial in range(N_TRIALS_GEN):
            shuffled = list(Z_problems)
            random.shuffle(shuffled)
            half = len(shuffled) // 2
            train_set = shuffled[:half]
            test_set = shuffled[half:]
            model_gen = SubtractionModel(current_input_dim, output_size=output_size)
            train(model_gen, train_set, mode, epochs=epochs_gen, verbose=False, result_min_val=result_min)
            correct = sum(1 for a,b,res in test_set if predict(model_gen,a,b,mode,result_min_val=result_min) == res)
            gen_accuracies.append(correct / len(test_set))
            if (trial+1) % 10 == 0:
                print(f"  Trial {trial+1}/{N_TRIALS_GEN}: acc = {gen_accuracies[-1]:.2%}")

        mean_acc = np.mean(gen_accuracies)
        std_acc = np.std(gen_accuracies)
        comparison_dict[mode] = {'mean': mean_acc, 'std': std_acc}
        print(f"\nGeneralisation accuracy: {mean_acc:.2%} ± {std_acc:.2%}")

        # ───────── HIDDEN SPACE VISUALIZATION ─────────
        print("\nVISUALIZING HIDDEN SPACE")

        def collect_representations(model, problems, m):
            reps, labels, results = [], [], []
            for a, b, result in problems:
                h = get_hidden(model, a, b, m)
                reps.append(h)
                results.append(result)
                labels.append('N' if result >= 0 else 'Z')
            return np.array(reps), labels, results

        reps_N_all, labels_N, results_N = collect_representations(model_N, Z_problems, mode)
        reps_Z_all, labels_Z, results_Z = collect_representations(model_Z, Z_problems, mode)

        pca = PCA(n_components=2)
        reps_N_2d = pca.fit_transform(reps_N_all)
        reps_Z_2d = pca.fit_transform(reps_Z_all)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, reps_2d, labels, results, title in [
            (axes[0], reps_N_2d, labels_N, results_N, f'model_N hidden space ({mode})'),
            (axes[1], reps_Z_2d, labels_Z, results_Z, f'model_Z hidden space ({mode})'),
        ]:
            colors = ['#1D9E75' if l == 'N' else '#7F77DD' for l in labels]
            ax.scatter(reps_2d[:, 0], reps_2d[:, 1], c=colors, alpha=0.7, s=60)
            for i, (x, y) in enumerate(reps_2d):
                ax.annotate(str(results[i]), (x, y), fontsize=7, ha='center', va='bottom', alpha=0.6)
            ax.set_title(title)
            ax.set_xlabel('PCA dimension 1')
            ax.set_ylabel('PCA dimension 2')
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color='#1D9E75', label='N problems'),
                               Patch(color='#7F77DD', label='Z only problems')])

        plt.suptitle(f'Internal space organization — {task_name} ({mode})', fontsize=13)
        plt.tight_layout()
        fname = f'{plot_dir}hidden_space_{task_name}_run{RUN_ID}.png'
        plt.savefig(fname, dpi=150)
        plt.clf()
        plt.close()
        print(f"Plot saved: {fname}")

        # ───────── ARC HYPOTHESIS ─────────
        pca2 = PCA(n_components=2)
        reps_Z_2d_v2 = pca2.fit_transform(reps_Z_all)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reps_Z_2d_v2[:, 0], reps_Z_2d_v2[:, 1],
                              c=results_Z, cmap='RdYlGn', s=80, alpha=0.8)
        plt.colorbar(scatter, label='result value')
        for i, (x, y) in enumerate(reps_Z_2d_v2):
            ax.annotate(str(results_Z[i]), (x, y), fontsize=7, ha='center', va='bottom')
        ax.set_title(f'model_Z hidden space colored by result ({task_name}, {mode})')
        ax.set_xlabel('PCA dimension 1')
        ax.set_ylabel('PCA dimension 2')
        plt.tight_layout()
        fname = f'{plot_dir}arc_hypothesis_{task_name}_run{RUN_ID}.png'
        plt.savefig(fname, dpi=150)
        plt.clf()
        plt.close()
        print(f"Plot saved: {fname}")

        # ───────── CORRELATION ANALYSIS ─────────
        print("\nCORRELATION ANALYSIS")
        target_result = 3 if task_name == 'subtraction' else 10
        result_vals = [(a, b, r) for a, b, r in Z_problems if r == target_result]
        print(f"\nAll problems with result = {target_result}:")
        for a, b, r in result_vals:
            h = get_hidden(model_Z, a, b, mode)
            h_2d = pca2.transform(h.reshape(1, -1))
            print(f"  {a} {'-' if task_name=='subtraction' else '+'} {b} = {r} | PCA: ({h_2d[0,0]:.2f}, {h_2d[0,1]:.2f}) | a={a}, b={b}, a+b={a+b}")

        all_reps_coords, all_a, all_b, all_results, all_aplusb = [], [], [], [], []
        for a, b, result in Z_problems:
            h = get_hidden(model_Z, a, b, mode)
            h_2d = pca2.transform(h.reshape(1, -1))
            all_reps_coords.append(h_2d[0])
            all_a.append(a); all_b.append(b); all_results.append(result); all_aplusb.append(a + b)

        pca1_coords = [r[0] for r in all_reps_coords]
        pca2_coords = [r[1] for r in all_reps_coords]

        print("\nLinear correlations:")
        print(f"  PCA dim 1 vs result:  {corrcoef(pca1_coords, all_results)[0,1]:.4f}")
        print(f"  PCA dim 1 vs a:       {corrcoef(pca1_coords, all_a)[0,1]:.4f}")
        print(f"  PCA dim 1 vs b:       {corrcoef(pca1_coords, all_b)[0,1]:.4f}")
        print(f"  PCA dim 1 vs a+b:     {corrcoef(pca1_coords, all_aplusb)[0,1]:.4f}")
        print()
        print(f"  PCA dim 2 vs result:  {corrcoef(pca2_coords, all_results)[0,1]:.4f}")
        print(f"  PCA dim 2 vs a:       {corrcoef(pca2_coords, all_a)[0,1]:.4f}")
        print(f"  PCA dim 2 vs b:       {corrcoef(pca2_coords, all_b)[0,1]:.4f}")
        print(f"  PCA dim 2 vs a+b:     {corrcoef(pca2_coords, all_aplusb)[0,1]:.4f}")

        all_axb = [a * b for a, b in zip(all_a, all_b)]
        all_absdiff = [abs(r) for r in all_results]

        print("\nNonlinear features vs PCA dim 2:")
        print(f"  vs |result|:  {corrcoef(pca2_coords, all_absdiff)[0,1]:.4f}")
        print(f"  vs a×b:       {corrcoef(pca2_coords, all_axb)[0,1]:.4f}")
        print(f"  vs |a|:       {corrcoef(pca2_coords, [abs(a) for a in all_a])[0,1]:.4f}")
        print(f"  vs |b|:       {corrcoef(pca2_coords, [abs(b) for b in all_b])[0,1]:.4f}")
        print(f"  vs max(a,b):  {corrcoef(pca2_coords, [max(a,b) for a,b in zip(all_a,all_b)])[0,1]:.4f}")
        print(f"  vs min(a,b):  {corrcoef(pca2_coords, [min(a,b) for a,b in zip(all_a,all_b)])[0,1]:.4f}")

        # ───────── THRESHOLD EXPERIMENT ─────────
        print("\n" + "=" * 50)
        print(f"THRESHOLD EXPERIMENT ({mode})")
        print("=" * 50)

        def threshold_experiment(all_problems, fractions, mode_str, epochs=500, trials=15, rmin=result_min):
            print(f"\n{'Train %':>10} | {'Train N':>8} | {'Test N':>7} | {'Mean Acc':>10} | {'Std Dev':>8}")
            print("-" * 62)
            results = []
            for frac in fractions:
                accs = []
                for _ in range(trials):
                    temp_list = list(all_problems)
                    random.shuffle(temp_list)
                    n_train = int(len(temp_list) * frac)
                    train_set = temp_list[:n_train]
                    test_set = temp_list[n_train:]
                    if not test_set: continue
                    model = SubtractionModel(get_input_dim(mode_str), output_size=output_size).to(DEVICE)
                    train(model, train_set, mode_str, epochs=epochs, verbose=False, result_min_val=rmin)
                    correct = sum(1 for a,b,res in test_set if predict(model,a,b,mode_str,result_min_val=rmin) == res)
                    accs.append(correct / len(test_set))
                accs_np = np.array(accs, dtype=np.float32)
                avg_acc = accs_np.mean()
                std_acc = accs_np.std(ddof=1) if len(accs) > 1 else 0.0
                n_train = int(len(all_problems) * frac)
                n_test = len(all_problems) - n_train
                print(f"{frac*100:>9.0f}% | {n_train:>8d} | {n_test:>7d} | {avg_acc:>9.1%} | {std_acc:>8.1%}")
                results.append((frac, avg_acc, std_acc))
            return results

        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresh_data = threshold_experiment(Z_problems, fractions, mode, rmin=result_min)

        fracs_plt = [r[0] * 100 for r in thresh_data]
        accs_plt  = [r[1] * 100 for r in thresh_data]

        plt.figure(figsize=(8, 5))
        plt.plot(fracs_plt, accs_plt, 'o-', color='#7F77DD', linewidth=2, markersize=8)
        plt.axhline(y=50, color='#888780', linestyle='--', alpha=0.5, label='50% line')
        plt.xlabel('% of problems used for training')
        plt.ylabel('generalization accuracy (%)')
        plt.title(f'Memorization vs Generalization — {task_name} ({mode})')
        plt.legend(); plt.grid(alpha=0.3)
        fname = f'{plot_dir}threshold_{task_name}_run{RUN_ID}.png'
        plt.savefig(fname, dpi=150)
        plt.clf()
        plt.close()
        print(f"\nPlot saved: {fname}")

        # ───────── OUTSIDE TRAINING RANGE TEST ─────────
        print("\nGENERALIZATION QUALITY TEST")
        model_gen_quality = SubtractionModel(current_input_dim, output_size=output_size)
        train(model_gen_quality, Z_problems, mode, epochs=epochs_gen, verbose=False, result_min_val=result_min)

        print("Testing OUTSIDE training range (model never saw these a,b values):")
        outside_cases = [(11, 5), (12, 7), (15, 3), (11, 11), (20, 10)]
        for a, b in outside_cases:
            p, conf = predict_with_confidence(model_gen_quality, a, b, mode, result_min_val=result_min)
            correct = a - b if task_name == 'subtraction' else a + b
            print(f"  {a} {'-' if task_name=='subtraction' else '+'} {b} = {p} (correct: {correct}) | confidence: {conf:.2%}")

        # ───────── SMOOTHNESS METRIC ─────────
        smooth = compute_smoothness(model_Z, Z_problems, mode, result_min_val=result_min)
        print(f"\nSMOOTHNESS (input–hidden distance correlation): {smooth:.4f}")

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
if 'subtraction' in args.tasks:
    run_task_experiments('subtraction', MIN_VAL, MAX_VAL, OUTPUT_SIZE,
                         comparison_stats_subtraction,
                         epochs_train=args.epochs_train,
                         epochs_gen=args.epochs_gen)

if 'addition' in args.tasks:
    run_task_experiments('addition', 0, 20, 21,
                         comparison_stats_addition,
                         epochs_train=args.epochs_train,
                         epochs_gen=args.epochs_gen)

# ─────────────────────────────────────────
# FINAL COMPARISON PLOT WITH ERROR BARS
# ─────────────────────────────────────────
def plot_comparison(stats_dict, task_name):
    if not stats_dict:
        return
    modes = list(stats_dict.keys())
    means = [stats_dict[m]['mean'] * 100 for m in modes]
    stds  = [stats_dict[m]['std']  * 100 for m in modes]
    plt.figure(figsize=(10, 6))
    plt.bar(modes, means, yerr=stds, capsize=5,
            color=['#1D9E75', '#7F77DD', '#E24B4A', '#F1C40F', '#8E44AD'])
    plt.ylabel('Generalisation Accuracy (%)')
    plt.title(f'{task_name.capitalize()} Task – Generalisation after 50/50 split\n'
              f'(mean ± std, {N_TRIALS_GEN} trials, seed={args.seed})')
    plt.ylim(0, 105)
    for i, (m, s) in enumerate(zip(means, stds)):
        plt.text(i, m + s + 2, f"{m:.1f}±{s:.1f}%", ha='center', fontweight='bold', fontsize=9)
    fname = f'encoder_comparison_{task_name}_final_run{RUN_ID}.png'
    plt.savefig(fname, dpi=150)
    plt.clf()
    plt.close()
    print(f"Comparison plot saved: {fname}")

plot_comparison(comparison_stats_subtraction, 'subtraction')
plot_comparison(comparison_stats_addition, 'addition')

print("\n" + "="*50)
print("ALL EXPERIMENTS COMPLETE.")
print("="*50)