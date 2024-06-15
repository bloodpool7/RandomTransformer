import math
import os 
import time 
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score

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
                 nlayers: int, input_size: int = 64, dropout: float = 0.5, averaging: bool = False):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.averaging = averaging
        if self.averaging:
            self.feedforward = nn.Sequential(
                                        nn.Linear(d_model, 7), 
                                        nn.Sigmoid())
        else:
            self.feedforward = nn.Sequential(
                                        nn.Flatten(), 
                                        nn.Linear(d_model * input_size, d_model * 16), 
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
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        output = self.transformer_encoder(src, src_mask)
        output = torch.mean(output, 1) if self.averaging else output
        output = self.feedforward(output)
        return output

class QueueDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.queues = tokenize(data['queue'].values)
        self.labels = torch.Tensor([list(map(int, list(x))) for x in data['label'].values])

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
        output = torch.cat((output, sixteen_bit_chunks))
    
    return output.reshape(-1, sequence_length)

def F_score(output, label, threshold=0.5): #Calculate the accuracy of the model
    prob = output > threshold
    label = label > threshold

    macro = f1_score(label, prob, average='macro', zero_division=0)
    micro = f1_score(label, prob, average='micro', zero_division=0)
    sample = f1_score(label, prob, average='samples', zero_division=0)
    weighted = f1_score(label, prob, average='weighted', zero_division=0)

    return macro, micro, sample, weighted

def train(transformer: nn.Module, criterion: nn, optimizer: torch.optim, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5, threshold: float = 0.5, device = 'cpu'):
    train_losses = []
    train_macro = []
    train_micro = []
    train_sample = []
    train_weighted = []

    val_losses = []
    val_macro = []
    val_micro = []
    val_sample = []
    val_weighted = []

    batch_num = []

    for epoch in range(epochs):
        transformer.train()

        train_losses_per_100 = []
        train_macro_per_100 = []
        train_micro_per_100 = []
        train_sample_per_100 = []
        train_weighted_per_100 = []

        val_losses_per_100 = []
        val_macro_per_100 = []
        val_micro_per_100 = []
        val_sample_per_100 = []
        val_weighted_per_100 = []

        for i, data in enumerate(train_loader):
            #get the data
            queues, labels = data
            queues = queues.to(device)
            labels = labels.to(device)

            #forward pass
            optimizer.zero_grad()
            output = transformer(queues)
            loss = criterion(output, labels)

            #backward pass
            loss.backward()
            optimizer.step()

            #metrics
            macro, micro, sample, weighted = F_score(output.to('cpu'), labels.to('cpu'), threshold)

            train_losses_per_100.append(loss.item())
            train_macro_per_100.append(macro)
            train_micro_per_100.append(micro)
            train_sample_per_100.append(sample)
            train_weighted_per_100.append(weighted)

            #every 100 batches, log metrics and run validation
            if (i+1) % 100 == 0:
                train_losses.append(np.array(train_losses_per_100).mean())
                train_macro.append(np.array(train_macro_per_100).mean())
                train_micro.append(np.array(train_micro_per_100).mean())
                train_sample.append(np.array(train_sample_per_100).mean())
                train_weighted.append(np.array(train_weighted_per_100).mean())

                train_losses_per_100 = []
                train_macro_per_100 = []
                train_micro_per_100 = []
                train_sample_per_100 = []
                train_weighted_per_100 = []

                batch_num.append(i + 1 + epoch * int(len(train_loader) / 100) * 100)

                #validate the model
                transformer.eval()
                for j, data in enumerate(val_loader):
                    queues, labels = data
                    queues = queues.to(device)
                    labels = labels.to(device)

                    #forward pass
                    output = transformer(queues)
                    loss = criterion(output, labels)

                    #metrics
                    macro, micro, sample, weighted = F_score(output.to('cpu'), labels.to('cpu'), threshold)

                    val_losses_per_100.append(loss.item())
                    val_macro_per_100.append(macro)
                    val_micro_per_100.append(micro)
                    val_sample_per_100.append(sample)
                    val_weighted_per_100.append(weighted)

                val_losses.append(np.array(val_losses_per_100).mean())
                val_macro.append(np.array(val_macro_per_100).mean())
                val_micro.append(np.array(val_micro_per_100).mean())
                val_sample.append(np.array(val_sample_per_100).mean())
                val_weighted.append(np.array(val_weighted_per_100).mean())
                
                val_losses_per_100 = []
                val_macro_per_100 = []
                val_micro_per_100 = []
                val_sample_per_100 = []
                val_weighted_per_100 = []
                
                print("epoch: {}, batch: {}, train loss: {:.3f}, train macro: {:.3f}, train micro: {:.3f}, train sample: {:.3f}, train weighted {:.3f}, val loss: {:.3f}, val macro: {:.3f}, val micro: {:.3f} val sample: {:.3f} val weighted: {:.3f}".format(
                    epoch + 1, i + 1, train_losses[-1], train_macro[-1], train_micro[-1], train_sample[-1], train_weighted[-1], val_losses[-1], val_macro[-1], val_micro[-1], val_sample[-1], val_weighted[-1])
                )
                transformer.train()
    
    train_dict = {}
    val_dict = {}

    train_dict["losses"] = train_losses
    train_dict["macro"] = train_macro
    train_dict["micro"] = train_micro
    train_dict["sample"] = train_sample
    train_dict["weighted"] = train_weighted

    val_dict["losses"] = val_losses
    val_dict["macro"] = val_macro
    val_dict["micro"] = val_micro
    val_dict["sample"] = val_sample
    val_dict["weighted"] = val_weighted

    train_df = pd.DataFrame(train_dict, index = batch_num)
    val_df = pd.DataFrame(val_dict, index = batch_num)

    del queues, labels, output, loss, transformer, criterion, optimizer

    return train_df, val_df

def inference(transformer: nn.Module, criterion: nn, test_loader: DataLoader, threshold:float = 0.5, device = 'cpu'):
    transformer.eval()
    test_losses = []
    micro_f1s = []
    macro_f1s = []
    sample_f1s = []
    weighted_f1s = []

    start = time.time()
    for i, data in enumerate(test_loader):
        queues, labels = data
        queues = queues.to(device)
        labels = labels.to(device)

        #forward pass
        output = transformer(queues)
        loss = criterion(output, labels)

        #metrics
        test_losses.append(loss.item())
        micro, macro, sample, weighted = F_score(output.to('cpu'), labels.to('cpu'), threshold)
        micro_f1s.append(micro)
        macro_f1s.append(macro)
        sample_f1s.append(sample)
        weighted_f1s.append(weighted)
    
    total_time = time.time() - start
    loss = np.array(test_losses).mean()
    micro = np.array(micro_f1s).mean()
    macro = np.array(macro_f1s).mean()
    sample = np.array(sample_f1s).mean()
    weighted = np.array(weighted_f1s).mean()
    
    print("Loss: {:.3f}".format(loss))
    print("Micro F1: {:.3f}".format(micro))
    print("Macro F1: {:.3f}".format(macro))
    print("Sample F1: {:.3f}".format(sample))
    print("Weighted F1: {:.3f}".format(weighted))
    print("Time: {:.3f}".format(total_time))

    return loss, micro, macro, sample, weighted, total_time

def plot_metrics(train_metrics: pd.DataFrame, val_metrics: pd.DataFrame):
    plt.figure(1)
    plt.plot(train_metrics.index, train_metrics['losses'], label = "Train Loss")
    plt.plot(val_metrics.index, val_metrics['losses'], label = "Validation Loss")
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(train_metrics.index, train_metrics['sample'], label = "Train Sample F1")
    plt.plot(val_metrics.index, val_metrics['sample'], label = "Validation Sample F1")
    plt.xlabel("Batch Number")
    plt.ylabel("Sample F1 Score")
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(train_metrics.index, train_metrics['macro'], label = "Train Macro F1")
    plt.plot(val_metrics.index, val_metrics['macro'], label = "Validation Macro F1")
    plt.xlabel("Batch Number")
    plt.ylabel("Macro F1 Score")
    plt.legend()
    plt.show()

    plt.figure(4)
    plt.plot(train_metrics.index, train_metrics['micro'], label = "Train Micro F1")
    plt.plot(val_metrics.index, val_metrics['micro'], label = "Validation Micro F1")
    plt.xlabel("Batch Number")
    plt.ylabel("Micro F1 Score")
    plt.legend()
    plt.show()

    plt.figure(5)
    plt.plot(train_metrics.index, train_metrics['weighted'], label = "Train Weighted F1")
    plt.plot(val_metrics.index, val_metrics['weighted'], label = "Validation Weighted F1")
    plt.xlabel("Batch Number")
    plt.ylabel("Weighted F1 Score")
    plt.legend()
    plt.show()

def model_save(transformer: nn.Module, path: str, train_metrics: pd.DataFrame = None, val_metrics: pd.DataFrame = None, test_metrics: tuple = None) -> None:
    torch.save(transformer, path + ".pt")

    train_metrics.to_csv(path + "_train_metrics.csv") if train_metrics is not None else None
    val_metrics.to_csv(path + "_val_metrics.csv") if val_metrics is not None else None

    with open(path + "_test_metrics.txt", "w") as file:
        file.write("Loss: {}\n".format(test_metrics[0]))
        file.write("Micro F1: {}\n".format(test_metrics[1]))
        file.write("Macro F1: {}\n".format(test_metrics[2]))
        file.write("Sample F1: {}\n".format(test_metrics[3]))
        file.write("Weighted F1: {}\n".format(test_metrics[4]))
        file.write("Time: {}\n".format(test_metrics[5]))

def visualize_exeperiment(path: str, tests: list[str], metric: str) -> None:
    # x axis will be the the tests 
    # y axis will be the metric 
    # one graph for averaging and one for non averaging

    #averaging graph 
    match path:
        case "EncoderResults/":
            x_values = [1, 2, 3, 4, 5, 6, 7, 8]
            x_label = "Encoder Layers"
        case "HeadsResults/":
            x_values = [1, 2, 4, 6, 8, 12, 16, 20, 24]
            x_label = "Self-attention Heads"
        case "EmbeddingsResults/":
            x_values = [144, 192, 240, 288, 336, 384, 432, 480, 528]
            x_label = "Embedding size"
    
    y_vals_512 = []
    y_vals_1024 = []
    y_vals_2048 = []

    for test in tests: 
        test_512 = path + "Averaging/" + test + "512_test_metrics.txt"
        test_1024 = path + "Averaging/" + test + "1024_test_metrics.txt"
        test_2048 = path + "Averaging/" + test + "2048_test_metrics.txt"

        with open(test_512, "r") as file:
            #get the specific metric out of loss micro f1 macro f1 sample f1 weighted f1 and time
            for line in file.readlines():
                if metric in line:
                    y_vals_512.append(float(line.split(":")[1]))
        
        with open(test_1024, "r") as file:
            for line in file.readlines():
                if metric in line:
                    y_vals_1024.append(float(line.split(":")[1]))
        
        with open(test_2048, "r") as file:
            for line in file.readlines():
                if metric in line:
                    y_vals_2048.append(float(line.split(":")[1]))
        
    plt.figure(1)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.plot(x_values, y_vals_512, label = "512")
    plt.plot(x_values, y_vals_1024, label = "1024")
    plt.plot(x_values, y_vals_2048, label = "2048")
    plt.xlabel(x_label, fontsize=19)
    plt.ylabel(metric, fontsize=19)
    plt.legend(fontsize = 16) 
    plt.show()

    y_vals_512 = []
    y_vals_1024 = []
    y_vals_2048 = []

    for test in tests: 
        test_512 = path + "NonAveraging/" + test + "512_test_metrics.txt"
        test_1024 = path + "NonAveraging/" + test + "1024_test_metrics.txt"
        test_2048 = path + "NonAveraging/" + test + "2048_test_metrics.txt"

        with open(test_512, "r") as file:
            #get the specific metric out of loss micro f1 macro f1 sample f1 weighted f1 and time
            for line in file.readlines():
                if metric in line:
                    y_vals_512.append(float(line.split(":")[1]))
        
        with open(test_1024, "r") as file:
            for line in file.readlines():
                if metric in line:
                    y_vals_1024.append(float(line.split(":")[1]))
        
        with open(test_2048, "r") as file:
            for line in file.readlines():
                if metric in line:
                    y_vals_2048.append(float(line.split(":")[1]))
        
    plt.figure(1)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.plot(x_values, y_vals_512, label = "512")
    plt.plot(x_values, y_vals_1024, label = "1024")
    plt.plot(x_values, y_vals_2048, label = "2048")
    plt.xlabel(x_label, fontsize=19)
    plt.ylabel(metric, fontsize=19)
    plt.legend(fontsize = 16) 
    plt.show()

