#!/usr/bin/env python3

# Import Libraries:
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import argparse, time, os, pandas as pd
from models import SentimentRNN
from utils import set_seeds, evaluate_model

# Function to Train a Model for One Epoch:
def train_model(model, train_loader, criterion, optimizer, clip_value=None, device='cpu',
                log_file=None, epoch_num=0):
    """Single epoch training loop with CSV batch logging"""
    model.train()
    epoch_loss = 0
    batch_logs = []

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader, 1):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
        optimizer.zero_grad()
        preds = model(x_batch).squeeze()
        loss = criterion(preds, y_batch)
        loss.backward()
        if clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        # Append Epoch Loss Data:
        batch_logs.append({
            "Epoch": epoch_num,
            "Batch": batch_idx,
            "Batch_Loss": batch_loss
        })

    # Write Batch Loss CSV for Each Model:
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        df_batch = pd.DataFrame(batch_logs)
        header = not os.path.exists(log_file)
        df_batch.to_csv(log_file, mode='a', header=header, index=False)

    return epoch_loss / len(train_loader)


def main():
    # Load Model Data from Driver:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed', help='Path to preprocessed data')
    parser.add_argument('--sequence_length', type=int, default=50)
    parser.add_argument('--architecture', type=str, default='RNN', choices=['RNN', 'LSTM', 'BiLSTM'])
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu', 'sigmoid'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('--clip', action='store_true', help='Apply gradient clipping')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--results_file', default='results/metrics.csv')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    args = parser.parse_args()

    # Validation: Confirm Sequence Length
    if args.sequence_length not in [25, 50, 100]:
        raise ValueError("Sequence length must be 25, 50, or 100")

    # Validation: Confirm if Activation is Compatible with LSTM/BiLSTM
    if args.architecture in ['LSTM', 'BiLSTM'] and args.activation != 'tanh':
        print(f"Warning: Activation argument '{args.activation}' is ignored for {args.architecture}.")

    # Utilities:
    set_seeds(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_total = time.time()

    # Load the Data:
    X_train = torch.load(os.path.join(args.data_dir, f'train_seq{args.sequence_length}.pt'))
    y_train = torch.load(os.path.join(args.data_dir, 'train_labels.pt'))
    X_test = torch.load(os.path.join(args.data_dir, f'test_seq{args.sequence_length}.pt'))
    y_test = torch.load(os.path.join(args.data_dir, 'test_labels.pt'))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    # Initiate the Model:
    model = SentimentRNN(architecture=args.architecture, activation=args.activation).to(device)
    criterion = nn.BCELoss()

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())
    else:
        raise ValueError("Unsupported optimizer")

    # Train the Model & Save Loss:
    print(f"\nRunning: {args.architecture}-{args.activation}-{args.optimizer}-seq{args.sequence_length}-clip{args.clip}")
    epoch_times, losses = [], []
    log_file = os.path.join("results", "batch_logs",
                            f"batchloss_{args.architecture}_{args.activation}_{args.optimizer}_seq{args.sequence_length}_clip{args.clip}.csv")

    for epoch in range(1, args.epochs + 1):
        start_epoch = time.time()
        clip_val = 0.5 if args.clip else None
        loss = train_model(model, train_loader, criterion, optimizer,
                           clip_value=clip_val, device=device,
                           log_file=log_file, epoch_num=epoch)
        end_epoch = time.time()
        losses.append(loss)
        epoch_times.append(end_epoch - start_epoch)
        print(f"Epoch {epoch}/{args.epochs} | Avg Loss: {loss:.4f} | Time: {end_epoch - start_epoch:.2f}s")

    # Evaluate the Model:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).squeeze().cpu()
            all_preds.append(preds)
            all_labels.append(y_batch)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    test_acc, f1 = evaluate_model(all_preds, all_labels)

    # Training Set Accuracy:
    all_train_preds = []
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).squeeze().cpu()
            all_train_preds.append(preds)
    all_train_preds = torch.cat(all_train_preds)
    train_acc, _ = evaluate_model(all_train_preds, y_train)

    print(f"\nFinal Results | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {f1:.4f}")
    print(f"Batch loss log saved to: {log_file}")

    # Save (or Update) Metrics CSV:
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    results = pd.DataFrame({
        "Architecture": [args.architecture],
        "Activation": [args.activation],
        "Optimizer": [args.optimizer],
        "Sequence_Length": [args.sequence_length],
        "Gradient_Clipping": [args.clip],
        "Train_Accuracy": [train_acc],
        "Test_Accuracy": [test_acc],
        "F1_Score": [f1],
        "Time_per_Epoch": [sum(epoch_times)/len(epoch_times)],
        "Total_Training_Time": [time.time() - start_total]
    })
    if os.path.exists(args.results_file):
        results.to_csv(args.results_file, mode='a', header=False, index=False)
    else:
        results.to_csv(args.results_file, index=False)

    # Save the Model for Later Reference:
    if args.save_model:
        model_dir = "results/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir,
                                  f"model_{args.architecture}_{args.sequence_length}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print(f"\nTotal Training Runtime: {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    main()
