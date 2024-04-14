import math
import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset

from sklearn.metrics import f1_score

# device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
device = "cpu"

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RandomLM(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.feedforward = nn.Sequential(
                                        nn.Flatten(), 
                                        nn.Linear(d_model * 64, d_model * 16), 
                                        nn.ReLU(), 
                                        nn.Linear(d_model * 16, d_model * 4), 
                                        nn.ReLU(), 
                                        nn.Linear(d_model * 4, 7), 
                                        nn.Sigmoid())

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.feedforward(output)
        return output

class QueueDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.queues = tokenize(data['queue'].values)
        self.labels = torch.Tensor([list(map(int, list(x))) for x in data['label'].values]).to(device)

    def __len__(self):
        return len(self.queues)

    def __getitem__(self, idx):
        return self.queues[idx], self.labels[idx]

def tokenize(data) -> Tensor:
    '''
    Tokenize the data into 16 bit chunks
    '''
    output = torch.LongTensor()
    sequence_length = 0
    for queue in data:
        sixteen_bit_chunks = [queue[i:i+16] for i in range(0, len(queue), 16)]
        sixteen_bit_chunks = np.array([int(i, 2) for i in sixteen_bit_chunks])
        sixteen_bit_chunks = torch.LongTensor(sixteen_bit_chunks)
        sequence_length = len(sixteen_bit_chunks)
        output = torch.cat((output, sixteen_bit_chunks)).to(device)
    
    return output.reshape(-1, sequence_length)

def F_score(output, label, threshold=0.5): #Calculate the accuracy of the model
    prob = output > threshold
    label = label > threshold

    macro = f1_score(label, prob, average='macro', zero_division=0)
    micro = f1_score(label, prob, average='micro', zero_division=0)
    sample = f1_score(label, prob, average='samples', zero_division=0)
    weighted = f1_score(label, prob, average='weighted', zero_division=0)

    return macro, micro, sample, weighted