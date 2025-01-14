# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:32:05 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

from torch import nn
import torch
import torch.nn.functional as F


# %%
class MultiHeadAttention(nn.Module):
    
    def __init__(self, hidden_size, num_heads):
        
        super(MultiHeadAttention, self).__init__()        
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):   # x: (batch_size, seq_len, hidden_size)        
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape Q, K, V for multi-head attention # (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim**0.5  
        attn_weights = F.softmax(scores, dim=-1)
        
        # Attention output
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)  # (batch_size, seq_len, hidden_size)
        
        output = self.out(attn_output)  # (batch_size, seq_len, hidden_size)        
        return output


class AttentionLSTM(nn.Module):
    
    def __init__(self, input_dimension, hidden_size, output_length, dropout_rate, num_heads):
        
        super(AttentionLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_dimension, hidden_size, num_layers=4, batch_first=True, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=4, batch_first=True, dropout=dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.fc = nn.Linear(hidden_size, output_length)
        
    def forward(self, x):   # x: (batch_size, seq_len, input_dimension)
        
        lstm_out1, _ = self.lstm1(x)  # lstm_out1: (batch_size, seq_len, hidden_size)
        lstm_out1 = self.ln1(lstm_out1)
        lstm_out1 = self.dropout1(lstm_out1)

        attn_out = self.attn(lstm_out1)  # (batch_size, seq_len, hidden_size)

        lstm_out2, _ = self.lstm2(attn_out)  # lstm_out2: (batch_size, seq_len, hidden_size)
        lstm_out2 = self.ln2(lstm_out2)
        lstm_out2 = self.dropout2(lstm_out2)
        
        prediction = self.fc(lstm_out2[:, -1, :])  # Only use the last time step for prediction (batch_size, output_length)
        return prediction


def create_model(data, forecast_horizon, device):
    
    input_dimension = len(data.columns)
    output_length = forecast_horizon
    
    hidden_size = 64
    dropout_rate = 0.05
    num_heads = 4
    
    model = AttentionLSTM(input_dimension, hidden_size, output_length, dropout_rate, num_heads).to(device)
    print("Attention LSTM model successfully initialized.")
    return model

