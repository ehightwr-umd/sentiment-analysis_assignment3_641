#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, argparse, os
from pathlib import Path

def plot_metrics(metrics_df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    # Accuracy vs Sequence Length
    plt.figure(figsize=(8,6))
    sns.lineplot(data=metrics_df, x='Sequence_Length', y='Test_Accuracy',
                 hue='Architecture', marker='o')
    plt.title('Test Accuracy vs Sequence Length')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Sequence Length')
    plt.ylim(0,1)
    plt.savefig(output_dir / 'accuracy_vs_seq_length.png')
    plt.close()

    # F1-score vs Sequence Length
    plt.figure(figsize=(8,6))
    sns.lineplot(data=metrics_df, x='Sequence_Length', y='F1_Score',
                 hue='Architecture', marker='o')
    plt.title('F1-Score vs Sequence Length')
    plt.ylabel('F1-Score')
    plt.xlabel('Sequence Length')
    plt.ylim(0,1)
    plt.savefig(output_dir / 'f1_vs_seq_length.png')
    plt.close()

    # Accuracy vs Architecture
    plt.figure(figsize=(8,6))
    sns.barplot(data=metrics_df, x='Architecture', y='Test_Accuracy', hue='Activation')
    plt.title('Test Accuracy per Architecture & Activation')
    plt.ylabel('Test Accuracy')
    plt.ylim(0,1)
    plt.savefig(output_dir / 'accuracy_vs_architecture.png')
    plt.close()

    # F1-score vs Architecture
    plt.figure(figsize=(8,6))
    sns.barplot(data=metrics_df, x='Architecture', y='F1_Score', hue='Activation')
    plt.title('F1-Score per Architecture & Activation')
    plt.ylabel('F1-Score')
    plt.ylim(0,1)
    plt.savefig(output_dir / 'f1_vs_architecture.png')
    plt.close()

    print(f"Metric plots saved in: {output_dir}")

def make_filename(row):
    """Constructs batchloss filename from row data."""
    return f"batchloss_{row.Architecture}_{row.Activation}_{row.Optimizer}_seq{row.Sequence_Length}_clip{row.Gradient_Clipping}.csv"

def make_configuration(row):
    """Constructs model configuration from row data."""
    return f"{row.Architecture}_{row.Activation}_{row.Optimizer}_seq{row.Sequence_Length}_clip{row.Gradient_Clipping}"

def plot_best_worst_loss(metrics_df, batchlog_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    # Identify Best & Worst: Accuracy & F1
    best_acc = metrics_df.loc[metrics_df['Test_Accuracy'].idxmax()]
    worst_acc = metrics_df.loc[metrics_df['Test_Accuracy'].idxmin()]
    best_f1 = metrics_df.loc[metrics_df['F1_Score'].idxmax()]
    worst_f1 = metrics_df.loc[metrics_df['F1_Score'].idxmin()]

    print("\nBest/Worst Models Identified:")
    print("Accuracy - Best Model", make_configuration(best_acc))
    print("Accuracy - Worst Model:", make_configuration(worst_acc))
    print("F1 - Best Model:", make_configuration(best_f1))
    print("F1 - Worst Model:", make_configuration(worst_f1))

    def load_loss_data(filename):
        file_path = batchlog_dir / filename
        if not file_path.exists():
            print(f"Missing batch log: {file_path}")
            return None
        df = pd.read_csv(file_path)
        loss_col = [c for c in df.columns if 'loss' in c.lower()][0]
        epoch_loss = df.groupby('Epoch')[loss_col].mean().reset_index()
        return epoch_loss

    # Fuction to Plot Best & Worst Performance (F1, Accuracy)
    def plot_loss(best_row, worst_row, metric_name):
        best_file = make_filename(best_row)
        worst_file = make_filename(worst_row)
        best_data = load_loss_data(best_file)
        worst_data = load_loss_data(worst_file)
        best_name = make_configuration(best_row)
        worst_name = make_configuration(worst_row)
        if best_data is None or worst_data is None:
            return

        plt.figure(figsize=(8,5))
        plt.plot(best_data['Epoch'], best_data.iloc[:,1], label=f'Best ({metric_name})', color='green')
        plt.plot(worst_data['Epoch'], worst_data.iloc[:,1], label=f'Worst ({metric_name})', color='blue')
        plt.title(f'{metric_name}: Training Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.suptitle(f"Best Model: {best_name}\nWorst Model: {worst_name}", fontsize=9, y=0.93, color='black')
        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Make space for subtitle
        plt.savefig(output_dir / f"training_loss_{metric_name.lower().replace(' ', '_')}.png")
        plt.close()
        print(f"Saved: training_loss_{metric_name.lower().replace(' ', '_')}.png")

    plot_loss(best_f1, worst_f1, "F1 Score")
    plot_loss(best_acc, worst_acc, "Accuracy")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_file', default='results/metrics.csv',
                        help='Path to metrics.csv (relative to project root)')
    parser.add_argument('--batchlog_dir', default='results/batch_logs',
                        help='Path to batch loss CSV directory (relative to project root)')
    parser.add_argument('--output_dir', default='results/plots',
                        help='Directory to save plots (relative to project root)')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    metrics_path = project_root / args.metrics_file
    batchlog_dir = project_root / args.batchlog_dir
    output_dir = project_root / args.output_dir

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    print(f"Loaded metrics: {metrics_df.shape[0]} experiments from {metrics_path}")

    plot_metrics(metrics_df, output_dir)
    plot_best_worst_loss(metrics_df, batchlog_dir, output_dir)

if __name__ == "__main__":
    main()
