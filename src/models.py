#!/usr/bin/env python3

# Import Libraries:
import torch
import torch.nn as nn
from utils import set_seeds

# Set Randomness, as required:
set_seeds(42)

# Define Model:
class SentimentRNN(nn.Module):
    def __init__(self, architecture='RNN', vocab_size=10000, embedding_dim=100,
                 hidden_size=64, num_layers=2, dropout=0.3, output_size=1,
                 activation='tanh'):
        super(SentimentRNN, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.architecture = architecture.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if activation not in ['tanh', 'relu', 'sigmoid']:
            raise ValueError("Activation must be 'tanh', 'relu', or 'sigmoid'")
        self.activation = activation

        # Architecture Selection: RNN
        if self.architecture == 'RNN':
            # Note: PyTorch RNN supports only 'tanh' or 'relu'
            if activation in ['tanh', 'relu']:
                nonlinearity = activation
                self.use_manual_sigmoid = False
            else:
                nonlinearity = 'tanh'
                self.use_manual_sigmoid = True
            # Special Function for 'sigmoid'
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

        # Architecture Selection: LSTM
        elif self.architecture == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.use_manual_sigmoid = False
            if activation != 'tanh':
                print(f"Warning: Activation argument '{activation}' is ignored for LSTM.")
        
        # Architecture Selection: BiLSTM
        elif self.architecture == 'BILSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            self.use_manual_sigmoid = False
            if activation != 'tanh':
                print(f"Warning: Activation argument '{activation}' is ignored for BiLSTM.")
        # Note: LSTM and BiLSTM only allow the 'tanh' activation function.
        # 'sigmoid' is already used in the three gates (input, forget, and output).
        # Activation with 'relu' or another arbitrary activation will break gating logic.
        # 'tanh' keeps the cell state stable as it scales between -1 and 1. It is used in the candidate cell state and final hidden output.
 
        else:
            raise ValueError("Architecture must be 'RNN', 'LSTM', or 'BiLSTM'")

        # Fully Connected Output:
        fc_input_size = hidden_size * 2 if self.architecture == 'BILSTM' else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)

        # Extract Hidden Last States
        if self.architecture == 'RNN':
            last_hidden = hidden[-1]
            if self.use_manual_sigmoid:
                last_hidden = torch.sigmoid(last_hidden)

        elif self.architecture in ['LSTM', 'BILSTM']:
            h_n = hidden[0]
            if self.architecture == 'BILSTM':
                # For BiLSTM, concatenate last forward and backward states.
                forward_final = h_n[-2, :, :]
                backward_final = h_n[-1, :, :]
                last_hidden = torch.cat((forward_final, backward_final), dim=1)
            else:
                last_hidden = h_n[-1]

        # Apply Dropout + Fully Connected
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        out = self.out_activation(out)
        return out.squeeze()
