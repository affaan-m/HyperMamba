# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:45:42 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch
from torch import nn


# %%
class Seq2SeqLSTM(nn.Module):
    
    def __init__(self, input_dimension, hidden_size, output_length, dropout_rate, num_layers):
        
        super(Seq2SeqLSTM, self).__init__()        
        self.output_length = output_length
        
        self.encoder_lstm = nn.LSTM(input_dimension, hidden_size, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        
        batch_size = X.size(0)
        
        # Encoder: Pass the input sequence
        _, (hidden, cell) = self.encoder_lstm(X)
        
        # Decoder: Initialize the decoder input
        decoder_input = torch.zeros(batch_size, 1, 1).to(X.device)  # Start token (all zeros)
        outputs = []
        
        # Decode for the forecast horizon
        for t in range(self.output_length):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            out = self.dropout(out)  # Apply dropout here
            pred = self.fc(out)  # Predict the next value
            outputs.append(pred.squeeze(1))  # Collect predictions
            decoder_input = pred  # Feed predicted value as input for the next step

        outputs = torch.stack(outputs, dim=1)  # Combine all outputs into a sequence
        return outputs


def create_model(data, forecast_horizon, device):
    
    input_dimension = len(data.columns)
    output_length = forecast_horizon
    
    hidden_size = 32
    dropout_rate = 0.05
    num_layers = 4
    
    model = Seq2SeqLSTM(input_dimension, hidden_size, output_length, dropout_rate, num_layers).to(device)
    print("Seq2Seq LSTM model successfully initialized.")
    return model

