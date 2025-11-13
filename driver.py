#!/usr/bin/env python3
import os, itertools, subprocess, pandas as pd, time

# Experiment (Model Combinations):
architectures = ["RNN", "LSTM", "BiLSTM"]
activations = ["sigmoid", "relu", "tanh"]
optimizers = ["adam", "sgd", "rmsprop"]
sequence_lengths = [25, 50, 100]
gradient_clipping = [False, True]

# Paths:
metrics_file = "results/metrics.csv"
plots_dir = "results/plots"
os.makedirs("results", exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Initialize Master Metrics CSV (if it doesn't exist):
if not os.path.exists(metrics_file):
    df = pd.DataFrame(columns=[
        "Architecture",
        "Activation",
        "Optimizer",
        "Sequence_Length",
        "Gradient_Clipping",
        "Train_Accuracy",
        "Test_Accuracy",
        "F1_Score",
        "Time_per_Epoch",
        "Total_Training_Time"
    ])
    df.to_csv(metrics_file, index=False)

# Function to Run Single Experiment:
def run_experiment(arch, act, opt, seq_len, clip):
    clip_flag = "--clip" if clip else ""
    cmd = [
        "python",
        "src/train.py",
        "--architecture", arch,
        "--activation", act,
        "--optimizer", opt,
        "--sequence_length", str(seq_len)]

    if clip:
        cmd.append("--clip")

    print(f"\nRunning: {arch}-{act}-{opt}-seq{seq_len}-clip{clip}")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        total_time = end_time - start_time
        print(result.stdout)

        df_metrics = pd.read_csv(metrics_file)
        last_row = df_metrics.iloc[-1]

        metrics = {
            "Train_Accuracy": last_row.get("Train_Accuracy", 0.0),
            "Test_Accuracy": last_row.get("Test_Accuracy", 0.0),
            "F1_Score": last_row.get("F1_Score", 0.0),
            "Time_per_Epoch": last_row.get("Time_per_Epoch", 0.0),
            "Total_Training_Time": total_time
        }
        return metrics

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed!\n{e.stderr}")
        return {
            "Train_Accuracy": 0.0,
            "Test_Accuracy": 0.0,
            "F1_Score": 0.0,
            "Time_per_Epoch": 0.0,
            "Total_Training_Time": time.time() - start_time
        }

# Function to Run All Experiments:
experiment_combinations = list(itertools.product(
    architectures, activations, optimizers, sequence_lengths, gradient_clipping
))

for arch, act, opt, seq_len, clip in experiment_combinations:
    df_existing = pd.read_csv(metrics_file)
    mask = (
        (df_existing["Architecture"] == arch) &
        (df_existing["Activation"] == act) &
        (df_existing["Optimizer"] == opt) &
        (df_existing["Sequence_Length"] == seq_len) &
        (df_existing["Gradient_Clipping"] == clip)
    )
    if mask.any():
        print(f"Skipping {arch}-{act}-{opt}-seq{seq_len}-clip{clip} (already completed).")
        continue

    metrics = run_experiment(arch, act, opt, seq_len, clip)
    new_row = pd.DataFrame([{
        "Architecture": arch,
        "Activation": act,
        "Optimizer": opt,
        "Sequence_Length": seq_len,
        "Gradient_Clipping": clip,
        **metrics
    }])
    df_updated = pd.concat([df_existing, new_row], ignore_index=True)
    df_updated.to_csv(metrics_file, index=False)


print(f"\nAll experiments done. Metrics saved to {metrics_file}")

# Run Evaluate.py:
try:
    print("\nGenerating summary plots...")
    cmd_eval = [
        "python",
        "src/evaluate.py",
        "--metrics_file", metrics_file,
        "--output_dir", plots_dir
    ]
    subprocess.run(cmd_eval, check=True)
    print(f"Summary plots saved in folder {plots_dir}")

except subprocess.CalledProcessError as e:
    print(f"ERROR: Failed to generate summary plots!\n{e.stderr}")
