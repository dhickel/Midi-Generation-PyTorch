import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Incept import IncepModule, IncepBottleNeckModule, IncepModule2, IncepModuleMidi
from IndRNN import MultiLayerIdnRNN


class EmbConvLstm(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstm, self).__init__()

        self.embedding_size = 64
        self.embedding_size_alt = int(self.embedding_size / 4)
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size_alt)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size_alt)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size_alt)

        self.e_size = self.embedding_size + (self.embedding_size_alt * 3)

        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv5 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=16, padding=0)
        self.relu = nn.LeakyReLU()

        self.attn = SelfAttention(self.e_size * 2 * 5)

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.e_size * 10).to(device)
        cell = torch.zeros(2, batch_size, self.e_size * 10).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)

        padded_embeddings5 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out5 = self.conv5(padded_embeddings5)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5), dim=1)

        conv_out = self.dropout(conv_out)

        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.attn(conv_out)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity, hidden


# class SelfAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#
#     def forward(self, x):
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#         weights = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1]), dim=-1)
#         return weights @ v


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.qkv = nn.Linear(input_dim, input_dim * 3)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        weights = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1]), dim=-1)
        return weights @ v


# class SelfAttentionNorm(nn.Module):
#     def __init__(self, input_dim):
#         super(SelfAttentionNorm, self).__init__()
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#         self.layernorm = nn.LayerNorm(input_dim)
#
#     def forward(self, x):
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#         weights = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1]), dim=-1)
#         return self.layernorm(weights @ v)
#

def positional_encoding(seq_len, d_model, device):
    PE = torch.zeros(seq_len, d_model).to(device)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(device)
    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE


class EmbConvLstmAttn(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttn, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches
        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        # field 7
        # self.conv3a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        # self.conv3b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)
        # field 4
        self.conv4 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv5 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        # field 16
        self.conv6a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv6b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=6, padding=0)

        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=16, padding=0)

        self.attention2 = SelfAttention(self.e_size * 6 * 2)

        # self.ln1 = nn.LayerNorm(self.e_size * 2 * 5)  # Change size as needed

        self.relu = nn.ReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 6 * 2, self.e_size * 6 * 2, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        # self.attention3 = SelfAttention(self.e_size * 4 + (self.e_size * 2))

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)

        # embeddings = self.attention1(embeddings)

        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        padded_embeddings1 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings1)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out4 = self.conv4(padded_embeddings3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings4)

        padded_embeddings5 = F.pad(embeddings, (4, 3))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out6 = self.conv6a(padded_embeddings5)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6b(conv_out6)

        padded_embeddings6 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out61 = self.conv6(padded_embeddings6)

        # conv_out6 = (conv_out6 + conv_out61) / 2.0

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out4, conv_out5, conv_out6, conv_out61), dim=1)

        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.attention2(conv_out)

        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)
        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]
        # lstm_out = self.attention3(lstm_out)
        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixed(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixed, self).__init__()

        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        # field 4
        self.conv4 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv5 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        # field 8
        self.conv6a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv6b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=6, padding=0)
        # field 16
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=16, padding=0)

        self.attention1 = SelfAttention(self.e_size * 6 * 2)
        self.layernorm = nn.LayerNorm(self.e_size * 6 * 2)
        self.relu = nn.Mish()

        self.lstm = nn.LSTM(self.e_size * 6 * 2, self.e_size * 6 * 2, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)
        #
        padded_embeddings1 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings1)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out4 = self.conv4(padded_embeddings3)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings4)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings5 = F.pad(embeddings, (4, 3))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out6 = self.conv6a(padded_embeddings5)
        conv_out6 = self.dropout(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6b(conv_out6)
        conv_out6 = self.relu(conv_out6)

        padded_embeddings6 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out61 = self.conv6(padded_embeddings6)
        conv_out61 = self.relu(conv_out61)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out4, conv_out5, conv_out6, conv_out61), dim=1)
        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        # lstm_out = self.attention2(lstm_out)
        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixed2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixed2, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        # self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)

        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)

        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=0)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        # field 4
        self.conv5a = nn.Conv1d(self.e_size, self.e_size, kernel_size=4, padding=0)
        self.conv5b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)

        self.attention1 = SelfAttention(self.e_size * 12)
        self.layernorm = nn.LayerNorm(self.e_size * 12)
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(self.e_size * 12, self.e_size * 12, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 12 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 12 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 12 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 12 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings1 = F.pad(embeddings, (1, 1))
        conv_out2 = self.conv2(padded_embeddings1)

        padded_embeddings2 = F.pad(embeddings, (2, 1))
        conv_out3 = self.conv3(padded_embeddings2)

        padded_embeddings3 = F.pad(embeddings, (2, 3))
        conv_out4 = self.conv4a(padded_embeddings3)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 3))
        conv_out5 = self.conv5a(padded_embeddings4)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out6 = self.conv6(padded_embeddings5)
        conv_out6 = self.relu(conv_out6)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixed3(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixed3, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        # self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv3a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=0)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv4c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)
        # field 4
        self.conv5a = nn.Conv1d(self.e_size, self.e_size, kernel_size=4, padding=0)
        self.conv5b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv5c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)
        # field 8
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv6a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.attention1 = SelfAttention(self.e_size * 12)
        self.layernorm = nn.LayerNorm(self.e_size * 12)
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(self.e_size * 12, self.e_size * 12, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 12 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 12 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 12 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 12 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings1 = F.pad(embeddings, (1, 1))
        conv_out2 = self.conv2(padded_embeddings1)
        conv_out2 = self.relu(self.dropout(conv_out2))
        conv_out2 = self.conv2a(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings2 = F.pad(embeddings, (2, 1))
        conv_out3 = self.conv3(padded_embeddings2)
        conv_out3 = self.relu(self.dropout(conv_out3))
        conv_out3 = self.conv3a(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings3 = F.pad(embeddings, (2, 3))
        conv_out4 = self.conv4a(padded_embeddings3)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4c(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 3))
        conv_out5 = self.conv5a(padded_embeddings4)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5c(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out6 = self.conv6(padded_embeddings5)
        conv_out6 = self.relu(self.dropout(conv_out6))
        conv_out6 = self.conv6a(conv_out6)
        conv_out6 = self.relu(conv_out6)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixedInc(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixedInc, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        # self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.conv3o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv3a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.conv4o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=0)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv4c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)
        # field 4
        self.conv5o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size, self.e_size, kernel_size=4, padding=0)
        self.conv5b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv5c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)
        # field 8
        self.conv6o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv6a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=0)
        self.maxpoola = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)

        self.attention1 = SelfAttention(self.e_size * 13)
        self.layernorm = nn.LayerNorm(self.e_size * 13)
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(self.e_size * 13, self.e_size * 13, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 13 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 13 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 13 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 13 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings1 = F.pad(embeddings, (1, 1))
        conv_out2 = self.conv2o(padded_embeddings1)
        conv_out2 = self.relu(self.dropout(conv_out2))
        conv_out2 = self.conv2(conv_out2)
        conv_out2 = self.relu(self.dropout(conv_out2))
        conv_out2 = self.conv2a(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings2 = F.pad(embeddings, (2, 1))
        conv_out3 = self.conv3o(padded_embeddings2)
        conv_out3 = self.relu(self.dropout(conv_out3))
        conv_out3 = self.conv3(conv_out3)
        conv_out3 = self.relu(self.dropout(conv_out3))
        conv_out3 = self.conv3a(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings3 = F.pad(embeddings, (2, 3))
        conv_out4 = self.conv4o(padded_embeddings3)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4a(conv_out4)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4c(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 3))
        conv_out5 = self.conv5o(padded_embeddings4)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5c(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out6 = self.conv6o(padded_embeddings5)
        conv_out6 = self.relu(self.dropout(conv_out6))
        conv_out6 = self.conv6(conv_out6)
        conv_out6 = self.relu(self.dropout(conv_out6))
        conv_out6 = self.conv6a(conv_out6)
        conv_out6 = self.relu(conv_out6)

        padded_embeddings6 = F.pad(embeddings, (2, 2))
        max_pool = self.maxpool(padded_embeddings6)
        max_poola = self.maxpool(max_pool)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6, max_poola), dim=1)

        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixedInc2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixedInc2, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        # self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)

        self.conv3o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)

        self.conv4o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=0)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)

        # field 4
        self.conv5o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size, self.e_size, kernel_size=4, padding=0)
        self.conv5b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)

        # field 8
        self.conv6o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=0)
        self.maxpoola = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)

        self.attention1 = SelfAttention(self.e_size * 13)
        self.layernorm = nn.LayerNorm(self.e_size * 13)
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(self.e_size * 13, self.e_size * 13, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 13 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 13 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 13 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 13 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings1 = F.pad(embeddings, (1, 1))
        conv_out2 = self.conv2o(padded_embeddings1)
        conv_out2 = self.relu(self.dropout(conv_out2))
        conv_out2 = self.conv2(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings2 = F.pad(embeddings, (2, 1))
        conv_out3 = self.conv3o(padded_embeddings2)
        conv_out3 = self.relu(self.dropout(conv_out3))
        conv_out3 = self.conv3(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings3 = F.pad(embeddings, (2, 3))
        conv_out4 = self.conv4o(padded_embeddings3)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4a(conv_out4)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 3))
        conv_out5 = self.conv5o(padded_embeddings4)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out6 = self.conv6o(padded_embeddings5)
        conv_out6 = self.relu(self.dropout(conv_out6))
        conv_out6 = self.conv6(conv_out6)
        conv_out6 = self.relu(conv_out6)

        padded_embeddings6 = F.pad(embeddings, (2, 2))
        max_pool = self.maxpool(padded_embeddings6)
        max_poola = self.maxpool(max_pool)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6, max_poola), dim=1)

        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnTest(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnTest, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches

        # "linear"
        self.conv1 = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)

        # odd conv
        self.conv23 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv25a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv25b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=1)
        self.conv27a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv27b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=1)
        self.conv27c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=1)

        # even conv
        self.conv32 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=2, padding=0)
        self.conv34 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv38 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)

        # field 16
        self.conv4 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=16, padding=0)

        self.attention = SelfAttention(self.e_size * 9)
        self.relu = nn.ReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 9, self.e_size * 9, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        # self.attention3 = SelfAttention(self.e_size * 4 + (self.e_size * 2))

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 9 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 9 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 9 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 9 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)  # no padding required for kernel_size=1

        conv_out2 = self.conv23(embeddings)
        conv_out21 = self.conv25a(embeddings)
        conv_out21 = self.relu(conv_out21)
        conv_out21 = self.conv25b(conv_out21)

        conv_out22 = self.conv27a(embeddings)
        conv_out22 = self.relu(conv_out22)
        conv_out22 = self.conv27b(conv_out22)
        conv_out22 = self.relu(conv_out22)
        conv_out22 = self.conv27c(conv_out22)

        conv_out2 = torch.cat((conv_out2, conv_out21, conv_out22), dim=1)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings1 = F.pad(embeddings, (0, 1))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv32(padded_embeddings1)
        padded_embeddings2 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out31 = self.conv34(padded_embeddings2)
        padded_embeddings3 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out32 = self.conv38(padded_embeddings3)

        conv_out3 = torch.cat((conv_out3, conv_out31, conv_out32), dim=1)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = self.relu(conv_out4)

        conv_out23 = (conv_out2 + conv_out3) / 2.0

        conv_out = torch.cat((conv_out1, conv_out23, conv_out4), dim=1)

        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)
        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnSimple(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstmAttnSimple, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches

        # "linear"
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv3a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=1)
        self.conv3b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv3c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=1)
        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=5, padding=2)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=5, padding=2)
        self.conv4c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=5, padding=2)

        self.attention = SelfAttention(self.e_size * 2 * 4)
        self.relu = nn.ReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 2 * 4, self.e_size * 2 * 4, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 2 * 4 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 2 * 4 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 2 * 4 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 2 * 4 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        conv1 = self.conv1(embeddings)
        conv2 = self.conv2(embeddings)

        conv3 = self.conv3a(embeddings)
        conv3 = self.relu(self.dropout(conv3))
        conv3 = self.conv3b(self.dropout(conv3))
        conv3 = self.relu(self.dropout(conv3))

        conv4 = self.conv3a(embeddings)
        conv4 = self.relu(self.dropout(conv4))
        conv4 = self.conv3b(self.dropout(conv4))
        conv4 = self.relu(self.dropout(conv4))

        conv_out = torch.cat((conv1, conv2, conv3, conv4), dim=1)

        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.attention(conv_out)

        conv_out = self.dropout(conv_out)
        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnInc(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnInc, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        self.conv2i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)
        # field 4
        self.conv3i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv4i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv4 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=8, padding=0)
        # field 18 complex
        self.conv5i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=3, padding=0)
        self.conv5b = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=6, padding=0)

        # field 16 long
        self.conv6i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv6 = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=16, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 16, self.e_size * 4, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 4)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 4, self.e_size * 8, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 8 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 8 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 8 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 8 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2i(padded_embeddings2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3i(padded_embeddings3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4i(padded_embeddings4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (4, 3))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out5 = self.conv5i(padded_embeddings5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out6 = self.conv6i(padded_embeddings6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6(conv_out6)
        conv_out6 = self.relu(conv_out6)

        # conv_out6 = (conv_out6 + conv_out61) / 2.0

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = self.dropout(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnInc2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnInc2, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        self.conv2i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)
        # field 4
        self.conv3i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv4i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv4 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=8, padding=0)
        # field 18 complex
        self.conv5i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=3, padding=0)
        self.conv5b = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=4, padding=0)

        # field 16 long
        self.conv6i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv6a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=4, padding=0)
        self.conv6b = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=4, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 16, self.e_size * 4, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 4)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 4, self.e_size * 8, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 8 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 8 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 8 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 8 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2i(padded_embeddings2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3i(padded_embeddings3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4i(padded_embeddings4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (2, 3))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out5 = self.conv5i(padded_embeddings5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings,
                                   (3, 3))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out6 = self.conv6i(padded_embeddings6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6a(conv_out6)
        conv_out6 = self.conv6b(conv_out6)
        conv_out6 = self.relu(conv_out6)

        # conv_out6 = (conv_out6 + conv_out61) / 2.0

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = self.dropout(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnsmall(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnsmall, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 4, kernel_size=1, padding=0)

        # field 12 complex
        self.conv5i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=3, padding=0)
        self.conv5b = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=4, padding=0)

        # field 16 complex
        self.conv6i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv6a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=4, padding=0)
        self.conv6b = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=4, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 12, self.e_size * 3, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 3)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 3, self.e_size * 6, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings5 = F.pad(embeddings, (2, 3))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out5 = self.conv5i(padded_embeddings5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings,
                                   (3, 3))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out6 = self.conv6i(padded_embeddings6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6a(conv_out6)
        conv_out6 = self.conv6b(conv_out6)
        conv_out6 = self.relu(conv_out6)

        # conv_out6 = (conv_out6 + conv_out61) / 2.0

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out5, conv_out6), dim=1)
        conv_out = self.dropout(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnAlt(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnAlt, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv11 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv12 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv1comp = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=1, padding=0)

        self.conv21a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv21b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)

        self.conv22a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv22b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=12, padding=0)

        self.conv31a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv31b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)

        self.conv32a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv32b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=16, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 12, self.e_size * 3, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 3)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 3, self.e_size * 6, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        conv_out11 = self.conv11(embeddings)  # no padding required for kernel_size=1

        added_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out12 = self.conv12(added_embeddings5)

        conv_out1 = torch.cat((conv_out11, conv_out12), dim=1)
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.conv1comp(conv_out1)
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.dropout(conv_out1)

        padded_embeddings21 = F.pad(embeddings, (1, 1))
        conv_out21 = self.conv21a(padded_embeddings21)
        conv_out21 = self.relu(conv_out21)
        conv_out21 = self.conv21b(conv_out21)

        padded_embeddings22 = F.pad(embeddings, (6, 5))
        conv_out22 = self.conv22a(padded_embeddings22)
        conv_out22 = self.relu(conv_out22)
        conv_out22 = self.conv22b(conv_out22)

        conv_out2 = torch.cat((conv_out21, conv_out22), dim=1)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.dropout(conv_out2)

        padded_embeddings31 = F.pad(embeddings, (1, 2))
        conv_out31 = self.conv31a(padded_embeddings31)
        conv_out31 = self.relu(conv_out31)
        conv_out31 = self.conv31b(conv_out31)

        padded_embeddings32 = F.pad(embeddings, (7, 8))
        conv_out32 = self.conv32a(padded_embeddings32)
        conv_out32 = self.relu(conv_out32)
        conv_out32 = self.conv32b(conv_out32)

        conv_out3 = torch.cat((conv_out31, conv_out32), dim=1)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.dropout(conv_out3)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnAlt(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnAlt, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv11 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv12 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv1comp = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=1, padding=0)

        self.conv21a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv21b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)

        self.conv22a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv22b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=12, padding=0)

        self.conv31a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv31b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)

        self.conv32a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv32b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=16, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 12, self.e_size * 3, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 3)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 3, self.e_size * 6, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        conv_out11 = self.conv11(embeddings)  # no padding required for kernel_size=1

        added_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out12 = self.conv12(added_embeddings5)

        conv_out1 = torch.cat((conv_out11, conv_out12), dim=1)
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.conv1comp(conv_out1)
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.dropout(conv_out1)

        padded_embeddings21 = F.pad(embeddings, (1, 1))
        conv_out21 = self.conv21a(padded_embeddings21)
        conv_out21 = self.relu(conv_out21)
        conv_out21 = self.conv21b(conv_out21)

        padded_embeddings22 = F.pad(embeddings, (6, 5))
        conv_out22 = self.conv22a(padded_embeddings22)
        conv_out22 = self.relu(conv_out22)
        conv_out22 = self.conv22b(conv_out22)

        conv_out2 = torch.cat((conv_out21, conv_out22), dim=1)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.dropout(conv_out2)

        padded_embeddings31 = F.pad(embeddings, (1, 2))
        conv_out31 = self.conv31a(padded_embeddings31)
        conv_out31 = self.relu(conv_out31)
        conv_out31 = self.conv31b(conv_out31)

        padded_embeddings32 = F.pad(embeddings, (7, 8))
        conv_out32 = self.conv32a(padded_embeddings32)
        conv_out32 = self.relu(conv_out32)
        conv_out32 = self.conv32b(conv_out32)

        conv_out3 = torch.cat((conv_out31, conv_out32), dim=1)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.dropout(conv_out3)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstm8(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstm8, self).__init__()

        self.embedding_size = 64
        self.embedding_size_alt = int(self.embedding_size / 4)
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size_alt)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size_alt)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size_alt)

        self.e_size = self.embedding_size + (self.embedding_size_alt * 3)

        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv5 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=16, padding=0)
        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)

        padded_embeddings5 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out5 = self.conv5(padded_embeddings5)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5), dim=1)

        conv_out = self.dropout(conv_out)

        conv_out = conv_out.permute(0, 2, 1)

        lstm_out, _ = self.lstm(conv_out)
        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        weights = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1]), dim=-1)
        return weights @ v


class SelfAttentionImpr(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionImpr, self).__init__()
        self.qkv = nn.Linear(input_dim, input_dim * 3)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        weights = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1]), dim=-1)
        return weights @ v


# class SelfAttentionNorm(nn.Module):
#     def __init__(self, input_dim):
#         super(SelfAttentionNorm, self).__init__()
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#         self.layernorm = nn.LayerNorm(input_dim)
#
#     def forward(self, x):
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#         weights = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1]), dim=-1)
#         return self.layernorm(weights @ v)


def positional_encoding(seq_len, d_model, device):
    PE = torch.zeros(seq_len, d_model).to(device)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(device)
    PE[:, 0::2] = torch.sin(position * div_term)
    PE[:, 1::2] = torch.cos(position * div_term)
    return PE


class EmbConvLstmAttn(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttn, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches
        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        # field 7
        # self.conv3a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        # self.conv3b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)
        # field 4
        self.conv4 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv5 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        # field 16
        self.conv6a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv6b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=6, padding=0)

        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=16, padding=0)

        self.attention2 = SelfAttention(self.e_size * 6 * 2)

        # self.ln1 = nn.LayerNorm(self.e_size * 2 * 5)  # Change size as needed

        self.relu = nn.ReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 6 * 2, self.e_size * 6 * 2, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        # self.attention3 = SelfAttention(self.e_size * 4 + (self.e_size * 2))

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)

        # embeddings = self.attention1(embeddings)

        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        padded_embeddings1 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings1)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out4 = self.conv4(padded_embeddings3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings4)

        padded_embeddings5 = F.pad(embeddings, (4, 3))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out6 = self.conv6a(padded_embeddings5)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6b(conv_out6)

        padded_embeddings6 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out61 = self.conv6(padded_embeddings6)

        # conv_out6 = (conv_out6 + conv_out61) / 2.0

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out4, conv_out5, conv_out6, conv_out61), dim=1)

        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.attention2(conv_out)

        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)
        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]
        # lstm_out = self.attention3(lstm_out)
        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixed(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixed, self).__init__()

        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        # field 4
        self.conv4 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv5 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)

        # field 16
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=16, padding=0)

        self.attention1 = SelfAttention(self.e_size * 5 * 2)
        self.layernorm = nn.LayerNorm(self.e_size * 5 * 2)
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(self.e_size * 5 * 2, self.e_size * 5 * 2, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)
        #
        padded_embeddings1 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings1)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out4 = self.conv4(padded_embeddings3)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings4)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out61 = self.conv6(padded_embeddings6)
        conv_out61 = self.relu(conv_out61)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out4, conv_out5, conv_out61), dim=1)
        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        # lstm_out = self.attention2(lstm_out)
        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixed2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixed2, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        # self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)

        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)

        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=0)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        # field 4
        self.conv5a = nn.Conv1d(self.e_size, self.e_size, kernel_size=4, padding=0)
        self.conv5b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)

        self.attention1 = SelfAttention(self.e_size * 12)
        self.layernorm = nn.LayerNorm(self.e_size * 12)
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(self.e_size * 12, self.e_size * 12, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 12 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 12 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 12 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 12 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings1 = F.pad(embeddings, (1, 1))
        conv_out2 = self.conv2(padded_embeddings1)

        padded_embeddings2 = F.pad(embeddings, (2, 1))
        conv_out3 = self.conv3(padded_embeddings2)

        padded_embeddings3 = F.pad(embeddings, (2, 3))
        conv_out4 = self.conv4a(padded_embeddings3)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 3))
        conv_out5 = self.conv5a(padded_embeddings4)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out6 = self.conv6(padded_embeddings5)
        conv_out6 = self.relu(conv_out6)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixed3(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixed3, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        # self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv3a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=0)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv4c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)
        # field 4
        self.conv5a = nn.Conv1d(self.e_size, self.e_size, kernel_size=4, padding=0)
        self.conv5b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv5c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)
        # field 8
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv6a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.attention1 = SelfAttention(self.e_size * 12)
        self.layernorm = nn.LayerNorm(self.e_size * 12)
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(self.e_size * 12, self.e_size * 12, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 12 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 12 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 12 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 12 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=256):
        hidden = torch.zeros(1, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(1, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings1 = F.pad(embeddings, (1, 1))
        conv_out2 = self.conv2(padded_embeddings1)
        conv_out2 = self.relu(self.dropout(conv_out2))
        conv_out2 = self.conv2a(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings2 = F.pad(embeddings, (2, 1))
        conv_out3 = self.conv3(padded_embeddings2)
        conv_out3 = self.relu(self.dropout(conv_out3))
        conv_out3 = self.conv3a(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings3 = F.pad(embeddings, (2, 3))
        conv_out4 = self.conv4a(padded_embeddings3)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4c(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 3))
        conv_out5 = self.conv5a(padded_embeddings4)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5c(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out6 = self.conv6(padded_embeddings5)
        conv_out6 = self.relu(self.dropout(conv_out6))
        conv_out6 = self.conv6a(conv_out6)
        conv_out6 = self.relu(conv_out6)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixedInc(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixedInc, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        # self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.conv3o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv3a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.conv4o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=0)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv4c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)
        # field 4
        self.conv5o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size, self.e_size, kernel_size=4, padding=0)
        self.conv5b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv5c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)
        # field 8
        self.conv6o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv6a = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=0)
        self.maxpoola = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)

        self.attention1 = SelfAttention(self.e_size * 13)
        self.layernorm = nn.LayerNorm(self.e_size * 13)
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(self.e_size * 13, self.e_size * 13, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 13 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 13 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 13 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 13 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings1 = F.pad(embeddings, (1, 1))
        conv_out2 = self.conv2o(padded_embeddings1)
        conv_out2 = self.relu(self.dropout(conv_out2))
        conv_out2 = self.conv2(conv_out2)
        conv_out2 = self.relu(self.dropout(conv_out2))
        conv_out2 = self.conv2a(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings2 = F.pad(embeddings, (2, 1))
        conv_out3 = self.conv3o(padded_embeddings2)
        conv_out3 = self.relu(self.dropout(conv_out3))
        conv_out3 = self.conv3(conv_out3)
        conv_out3 = self.relu(self.dropout(conv_out3))
        conv_out3 = self.conv3a(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings3 = F.pad(embeddings, (2, 3))
        conv_out4 = self.conv4o(padded_embeddings3)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4a(conv_out4)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4c(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 3))
        conv_out5 = self.conv5o(padded_embeddings4)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5c(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out6 = self.conv6o(padded_embeddings5)
        conv_out6 = self.relu(self.dropout(conv_out6))
        conv_out6 = self.conv6(conv_out6)
        conv_out6 = self.relu(self.dropout(conv_out6))
        conv_out6 = self.conv6a(conv_out6)
        conv_out6 = self.relu(conv_out6)

        padded_embeddings6 = F.pad(embeddings, (2, 2))
        max_pool = self.maxpool(padded_embeddings6)
        max_poola = self.maxpool(max_pool)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6, max_poola), dim=1)

        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnFixedInc2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnFixedInc2, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        # self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)
        self.conv2o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=0)

        self.conv3o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)

        self.conv4o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=0)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)

        # field 4
        self.conv5o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size, self.e_size, kernel_size=4, padding=0)
        self.conv5b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)

        # field 8
        self.conv6o = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)
        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=0)
        self.maxpoola = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)

        self.attention1 = SelfAttention(self.e_size * 13)
        self.layernorm = nn.LayerNorm(self.e_size * 13)
        self.relu = nn.LeakyReLU()

        self.lstm = nn.LSTM(self.e_size * 13, self.e_size * 13, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 13 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 13 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 13 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 13 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)

        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings1 = F.pad(embeddings, (1, 1))
        conv_out2 = self.conv2o(padded_embeddings1)
        conv_out2 = self.relu(self.dropout(conv_out2))
        conv_out2 = self.conv2(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings2 = F.pad(embeddings, (2, 1))
        conv_out3 = self.conv3o(padded_embeddings2)
        conv_out3 = self.relu(self.dropout(conv_out3))
        conv_out3 = self.conv3(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings3 = F.pad(embeddings, (2, 3))
        conv_out4 = self.conv4o(padded_embeddings3)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4a(conv_out4)
        conv_out4 = self.relu(self.dropout(conv_out4))
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings4 = F.pad(embeddings, (3, 3))
        conv_out5 = self.conv5o(padded_embeddings4)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(self.dropout(conv_out5))
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out6 = self.conv6o(padded_embeddings5)
        conv_out6 = self.relu(self.dropout(conv_out6))
        conv_out6 = self.conv6(conv_out6)
        conv_out6 = self.relu(conv_out6)

        padded_embeddings6 = F.pad(embeddings, (2, 2))
        max_pool = self.maxpool(padded_embeddings6)
        max_poola = self.maxpool(max_pool)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6, max_poola), dim=1)

        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.layernorm(conv_out)
        conv_out = self.attention1(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnTest(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnTest, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches

        # "linear"
        self.conv1 = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)

        # odd conv
        self.conv23 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv25a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv25b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=1)
        self.conv27a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv27b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=1)
        self.conv27c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=1)

        # even conv
        self.conv32 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=2, padding=0)
        self.conv34 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv38 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)

        # field 16
        self.conv4 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=16, padding=0)

        self.attention = SelfAttention(self.e_size * 9)
        self.relu = nn.ReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 9, self.e_size * 9, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        # self.attention3 = SelfAttention(self.e_size * 4 + (self.e_size * 2))

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 9 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 9 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 9 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 9 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)  # no padding required for kernel_size=1

        conv_out2 = self.conv23(embeddings)
        conv_out21 = self.conv25a(embeddings)
        conv_out21 = self.relu(conv_out21)
        conv_out21 = self.conv25b(conv_out21)

        conv_out22 = self.conv27a(embeddings)
        conv_out22 = self.relu(conv_out22)
        conv_out22 = self.conv27b(conv_out22)
        conv_out22 = self.relu(conv_out22)
        conv_out22 = self.conv27c(conv_out22)

        conv_out2 = torch.cat((conv_out2, conv_out21, conv_out22), dim=1)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings1 = F.pad(embeddings, (0, 1))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv32(padded_embeddings1)
        padded_embeddings2 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out31 = self.conv34(padded_embeddings2)
        padded_embeddings3 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out32 = self.conv38(padded_embeddings3)

        conv_out3 = torch.cat((conv_out3, conv_out31, conv_out32), dim=1)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = self.relu(conv_out4)

        conv_out23 = (conv_out2 + conv_out3) / 2.0

        conv_out = torch.cat((conv_out1, conv_out23, conv_out4), dim=1)

        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)
        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnSimple(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstmAttnSimple, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches

        # "linear"
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv3a = nn.Conv1d(self.e_size, self.e_size, kernel_size=3, padding=1)
        self.conv3b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=3, padding=1)
        self.conv3c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=1)
        self.conv4a = nn.Conv1d(self.e_size, self.e_size, kernel_size=5, padding=2)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=5, padding=2)
        self.conv4c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=5, padding=2)

        self.attention = SelfAttention(self.e_size * 2 * 4)
        self.relu = nn.ReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 2 * 4, self.e_size * 2 * 4, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 2 * 4 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 2 * 4 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 2 * 4 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 2 * 4 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        conv1 = self.conv1(embeddings)
        conv2 = self.conv2(embeddings)

        conv3 = self.conv3a(embeddings)
        conv3 = self.relu(self.dropout(conv3))
        conv3 = self.conv3b(self.dropout(conv3))
        conv3 = self.relu(self.dropout(conv3))

        conv4 = self.conv3a(embeddings)
        conv4 = self.relu(self.dropout(conv4))
        conv4 = self.conv3b(self.dropout(conv4))
        conv4 = self.relu(self.dropout(conv4))

        conv_out = torch.cat((conv1, conv2, conv3, conv4), dim=1)

        conv_out = conv_out.permute(0, 2, 1)

        conv_out = self.attention(conv_out)

        conv_out = self.dropout(conv_out)
        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnInc(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnInc, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        self.conv2i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)
        # field 4
        self.conv3i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv4i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv4 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=8, padding=0)
        # field 18 complex
        self.conv5i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=3, padding=0)
        self.conv5b = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=6, padding=0)

        # field 16 long
        self.conv6i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv6 = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=16, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 16, self.e_size * 4, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 4)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 4, self.e_size * 8, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 8 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 8 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 8 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 8 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2i(padded_embeddings2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3i(padded_embeddings3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4i(padded_embeddings4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (4, 3))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out5 = self.conv5i(padded_embeddings5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings,
                                   (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out6 = self.conv6i(padded_embeddings6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6(conv_out6)
        conv_out6 = self.relu(conv_out6)

        # conv_out6 = (conv_out6 + conv_out61) / 2.0

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = self.dropout(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnInc2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnInc2, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # field 3
        self.conv2i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)
        # field 4
        self.conv3i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        # field 8
        self.conv4i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv4 = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=8, padding=0)
        # field 18 complex
        self.conv5i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=3, padding=0)
        self.conv5b = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=4, padding=0)

        # field 16 long
        self.conv6i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv6a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=4, padding=0)
        self.conv6b = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=4, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 16, self.e_size * 4, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 4)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 4, self.e_size * 8, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 8 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 8 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 8 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 8 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2i(padded_embeddings2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3i(padded_embeddings3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4i(padded_embeddings4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (2, 3))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out5 = self.conv5i(padded_embeddings5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings,
                                   (3, 3))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out6 = self.conv6i(padded_embeddings6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6a(conv_out6)
        conv_out6 = self.conv6b(conv_out6)
        conv_out6 = self.relu(conv_out6)

        # conv_out6 = (conv_out6 + conv_out61) / 2.0

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = self.dropout(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnsmall(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnsmall, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 4, kernel_size=1, padding=0)

        # field 12 complex
        self.conv5i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv5a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=3, padding=0)
        self.conv5b = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=4, padding=0)

        # field 16 complex
        self.conv6i = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv6a = nn.Conv1d(self.e_size * 2, self.e_size * 4, kernel_size=4, padding=0)
        self.conv6b = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=4, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 12, self.e_size * 3, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 3)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 3, self.e_size * 6, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        padded_embeddings5 = F.pad(embeddings, (2, 3))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out5 = self.conv5i(padded_embeddings5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5a(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings,
                                   (3, 3))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out6 = self.conv6i(padded_embeddings6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6a(conv_out6)
        conv_out6 = self.conv6b(conv_out6)
        conv_out6 = self.relu(conv_out6)

        # conv_out6 = (conv_out6 + conv_out61) / 2.0

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out5, conv_out6), dim=1)
        conv_out = self.dropout(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnAlt(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnAlt, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv11 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv12 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv1comp = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=1, padding=0)

        self.conv21a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv21b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)

        self.conv22a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv22b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=12, padding=0)

        self.conv31a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv31b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)

        self.conv32a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv32b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=16, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 12, self.e_size * 3, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 3)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 3, self.e_size * 6, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        conv_out11 = self.conv11(embeddings)  # no padding required for kernel_size=1

        added_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out12 = self.conv12(added_embeddings5)

        conv_out1 = torch.cat((conv_out11, conv_out12), dim=1)
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.conv1comp(conv_out1)
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.dropout(conv_out1)

        padded_embeddings21 = F.pad(embeddings, (1, 1))
        conv_out21 = self.conv21a(padded_embeddings21)
        conv_out21 = self.relu(conv_out21)
        conv_out21 = self.conv21b(conv_out21)

        padded_embeddings22 = F.pad(embeddings, (6, 5))
        conv_out22 = self.conv22a(padded_embeddings22)
        conv_out22 = self.relu(conv_out22)
        conv_out22 = self.conv22b(conv_out22)

        conv_out2 = torch.cat((conv_out21, conv_out22), dim=1)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.dropout(conv_out2)

        padded_embeddings31 = F.pad(embeddings, (1, 2))
        conv_out31 = self.conv31a(padded_embeddings31)
        conv_out31 = self.relu(conv_out31)
        conv_out31 = self.conv31b(conv_out31)

        padded_embeddings32 = F.pad(embeddings, (7, 8))
        conv_out32 = self.conv32a(padded_embeddings32)
        conv_out32 = self.relu(conv_out32)
        conv_out32 = self.conv32b(conv_out32)

        conv_out3 = torch.cat((conv_out31, conv_out32), dim=1)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.dropout(conv_out3)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnAlt(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnAlt, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv11 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv12 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=8, padding=0)
        self.conv1comp = nn.Conv1d(self.e_size * 4, self.e_size * 4, kernel_size=1, padding=0)

        self.conv21a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv21b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)

        self.conv22a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv22b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=12, padding=0)

        self.conv31a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv31b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)

        self.conv32a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv32b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=16, padding=0)

        self.convcomp = nn.Conv1d(self.e_size * 12, self.e_size * 3, kernel_size=1, padding=0)
        self.attention = SelfAttentionImpr(self.e_size * 3)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 3, self.e_size * 6, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        conv_out11 = self.conv11(embeddings)  # no padding required for kernel_size=1

        added_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out12 = self.conv12(added_embeddings5)

        conv_out1 = torch.cat((conv_out11, conv_out12), dim=1)
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.conv1comp(conv_out1)
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.dropout(conv_out1)

        padded_embeddings21 = F.pad(embeddings, (1, 1))
        conv_out21 = self.conv21a(padded_embeddings21)
        conv_out21 = self.relu(conv_out21)
        conv_out21 = self.conv21b(conv_out21)

        padded_embeddings22 = F.pad(embeddings, (6, 5))
        conv_out22 = self.conv22a(padded_embeddings22)
        conv_out22 = self.relu(conv_out22)
        conv_out22 = self.conv22b(conv_out22)

        conv_out2 = torch.cat((conv_out21, conv_out22), dim=1)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.dropout(conv_out2)

        padded_embeddings31 = F.pad(embeddings, (1, 2))
        conv_out31 = self.conv31a(padded_embeddings31)
        conv_out31 = self.relu(conv_out31)
        conv_out31 = self.conv31b(conv_out31)

        padded_embeddings32 = F.pad(embeddings, (7, 8))
        conv_out32 = self.conv32a(padded_embeddings32)
        conv_out32 = self.relu(conv_out32)
        conv_out32 = self.conv32b(conv_out32)

        conv_out3 = torch.cat((conv_out31, conv_out32), dim=1)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.dropout(conv_out3)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnAlt2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnAlt2, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.note_data.n_vocab, kernel_size=1, padding=0)
        self.conv1a = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=1, padding=0)

        self.conv1b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=8, padding=0)

        # self.conv21a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv21b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=3, padding=0)

        # # self.conv22a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # self.conv22b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=12, padding=0)

        # self.conv31a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv31b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=4, padding=0)

        # # self.conv32a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        # self.conv32b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=16, padding=0)

        self.attention = SelfAttentionImpr(self.e_size * 4)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 4, self.e_size * 4, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 4 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 4 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 4 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 4 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        conv_out1i = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1i = self.relu(conv_out1i)  # no padding required for kernel_size=1

        # added_embeddings5 = F.pad(embeddings, (3, 4))
        conv_out11 = self.conv1a(conv_out1i)
        conv_out12 = self.conv1b(F.pad(conv_out1i, (3, 4)))

        conv_out21 = self.conv21b(F.pad(conv_out1i, (1, 1)))
        # conv_out22 = self.conv22b(F.pad(conv_out1i, (5, 6)))

        conv_out31 = self.conv31b(F.pad(conv_out1i, (1, 2)))
        #        conv_out32 = self.conv32b(F.pad(conv_out1i, (7, 8)))

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out11, conv_out21, conv_out31, conv_out12,), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnAlt3(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnAlt3, self).__init__()

        self.embedding_size1 = 96
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size, kernel_size=1, padding=0)

        self.conv3a = nn.Conv1d(self.e_size, self.note_data.n_vocab, kernel_size=1, padding=0)
        self.conv3b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=3, padding=0)

        self.conv4a = nn.Conv1d(self.e_size, self.note_data.n_vocab, kernel_size=1, padding=0)
        self.conv4b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=4, padding=0)

        self.conv8a = nn.Conv1d(self.e_size, self.note_data.n_vocab, kernel_size=1, padding=0)
        self.conv8b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=8, padding=0)

        # self.conv12a = nn.Conv1d(self.e_size, self.note_data.n_vocab, kernel_size=1, padding=0)
        # self.conv12b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=12, padding=0)

        self.conv16a = nn.Conv1d(self.e_size, self.note_data.n_vocab, kernel_size=1, padding=0)
        self.conv16b = nn.Conv1d(self.note_data.n_vocab, self.e_size, kernel_size=16, padding=0)

        self.attention = SelfAttentionImpr(self.e_size * 5)

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 5, self.e_size * 5, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_conn = nn.Linear(self.e_size * 5 * num_directions, self.e_size)
        self.fc_note = nn.Linear(self.e_size * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)

        conv_out3 = self.conv3a(F.pad(embeddings, (1, 1)))
        conv_out3 = self.conv3b(self.relu(conv_out3))
        conv_out3 = self.relu(conv_out3)

        conv_out4 = self.conv4a(F.pad(embeddings, (1, 2)))
        conv_out4 = self.conv4b(self.relu(conv_out4))
        conv_out4 = self.relu(conv_out4)

        conv_out8 = self.conv8a(F.pad(embeddings, (3, 4)))
        conv_out8 = self.conv8b(self.relu(conv_out8))
        conv_out8 = self.relu(conv_out8)

        # conv_out12 = self.conv12a(F.pad(embeddings, (5, 6)))
        # conv_out12 = self.conv12b(self.relu(conv_out12))
        # conv_out12 = self.relu(conv_out12)

        conv_out16 = self.conv16a(F.pad(embeddings, (7, 8)))
        conv_out16 = self.conv16b(self.relu(conv_out16))
        conv_out16 = self.relu(conv_out16)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out8, conv_out3, conv_out4, conv_out16), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)
        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.fc_conn(lstm_out)
        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnImpv(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnImpv, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches
        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)

        # field 3
        self.conv2 = nn.Conv1d(self.e_size, self.e * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size * 2, self.e * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.e_size * 2, self.e * 2, kernel_size=1, padding=0)

        # field 4
        self.conv3a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv3b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=2, padding=0)
        self.conv3c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)

        # field 8
        self.conv4a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv4b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=2, padding=0)
        self.conv4c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        self.conv4d = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)

        # field12
        self.conv5a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv5b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)
        self.conv5c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=5, padding=0)
        self.conv5d = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=6, padding=0)

        # field 16
        self.conv5a = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)
        self.conv5b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=2, padding=0)
        self.conv5c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=2, padding=0)
        self.conv5d = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        self.conv5e = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        self.conv5f = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=8, padding=0)

        self.attention2 = SelfAttention(self.e_size * 6 * 2)

        # self.ln1 = nn.LayerNorm(self.e_size * 2 * 5)  # Change size as needed

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 6 * 2, self.e_size * 6 * 2, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        # self.attention3 = SelfAttention(self.e_size * 4 + (self.e_size * 2))

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)

        # embeddings = self.attention1(embeddings)

        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2a(padded_embeddings2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2b(conv_out2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2c(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3a(padded_embeddings3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3b(conv_out3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3c(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4a(padded_embeddings4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4c(conv_out4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4d(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (5, 6))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5a(padded_embeddings5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5c(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5d(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings, (7, 8))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out6 = self.conv6a(padded_embeddings6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6b(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6c(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6d(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6e(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6f(conv_out6)
        conv_out6 = self.relu(conv_out6)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention2(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnImpv2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnImpv2, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.i = self.embedding_size1

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches
        # field 1
        self.conv1a = nn.Conv1d(self.e_size, self.i, kernel_size=1, padding=0)
        self.conv1b = nn.Conv1d(self.i, self.i * 2, kernel_size=1, padding=0)

        # field 3
        self.conv2a = nn.Conv1d(self.e_size, self.i, kernel_size=1, padding=0)
        self.conv2b = nn.Conv1d(self.i, self.i * 2, kernel_size=2, padding=0)
        self.conv2c = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=2, padding=0)

        # field 4
        self.conv3a = nn.Conv1d(self.e_size, self.i, kernel_size=1, padding=0)
        self.conv3b = nn.Conv1d(self.i, self.i * 2, kernel_size=2, padding=0)
        self.conv3c = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=3, padding=0)

        # field 8
        self.conv4a = nn.Conv1d(self.e_size, self.i, kernel_size=1, padding=0)
        self.conv4b = nn.Conv1d(self.i, self.i * 2, kernel_size=2, padding=0)
        self.conv4c = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=4, padding=0)
        self.conv4d = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=4, padding=0)

        # field12
        self.conv5a = nn.Conv1d(self.e_size, self.i, kernel_size=1, padding=0)
        self.conv5b = nn.Conv1d(self.i, self.i * 2, kernel_size=3, padding=0)
        self.conv5c = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=5, padding=0)
        self.conv5d = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=6, padding=0)

        # field 16
        self.conv6a = nn.Conv1d(self.e_size, self.i, kernel_size=1, padding=0)
        self.conv6b = nn.Conv1d(self.i, self.i * 2, kernel_size=6, padding=0)
        self.conv6c = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=6, padding=0)
        self.conv6d = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=6, padding=0)

        self.attention2 = SelfAttention(self.i * 6 * 2)

        # self.ln1 = nn.LayerNorm(self.e_size * 2 * 5)  # Change size as needed

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.i * 6 * 2, self.i * 6 * 2, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        # self.attention3 = SelfAttention(self.e_size * 4 + (self.e_size * 2))

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.i * 6 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.i * 6 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.i * 6 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.i * 6 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)

        # embeddings = self.attention1(embeddings)

        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1a(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.conv1b(conv_out1)  # no padding required for kernel_size=1

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2a(padded_embeddings2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2b(conv_out2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2c(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3a(padded_embeddings3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3b(conv_out3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3c(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4a(padded_embeddings4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4c(conv_out4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4d(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (5, 6))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5a(padded_embeddings5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5c(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5d(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings, (7, 8))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out6 = self.conv6a(padded_embeddings6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6b(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6c(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6d(conv_out6)
        conv_out6 = self.relu(conv_out6)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention2(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMWithAttention, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttention(hidden_dim)  # Replace with your attention implementation
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm1(x)
        output = self.attention(output)
        output, _ = self.lstm2(output)
        return output


class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMWithAttention, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttention(hidden_dim)  # Replace with your attention implementation
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        output, hidden = self.lstm1(x)
        output = self.attention(output)
        output, hidden = self.lstm2(output)
        return output, hidden


class EmbConvLstmAttnImpv22(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnImpv22, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.i = self.embedding_size1
        self.fc_embedding = nn.Linear(self.e_size, self.i)
        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches
        # field 1
        self.conv1a = nn.Conv1d(self.i, self.i * 2, kernel_size=1, padding=0)
        self.conv1b = nn.Conv1d(self.i * 2, self.i * 4, kernel_size=1, padding=0)

        # field 3
        self.conv2a = nn.Conv1d(self.i, self.i * 2, kernel_size=1, padding=0)
        self.conv2b = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=2, padding=0)
        self.conv2c = nn.Conv1d(self.i * 2, self.i * 4, kernel_size=2, padding=0)

        # field 4
        self.conv3a = nn.Conv1d(self.i, self.i * 2, kernel_size=1, padding=0)
        self.conv3b = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=2, padding=0)
        self.conv3c = nn.Conv1d(self.i * 2, self.i * 4, kernel_size=3, padding=0)

        # field 8
        self.conv4a = nn.Conv1d(self.i, self.i * 2, kernel_size=1, padding=0)
        self.conv4b = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=2, padding=0)
        self.conv4c = nn.Conv1d(self.i * 2, self.i * 4, kernel_size=4, padding=0)
        self.conv4d = nn.Conv1d(self.i * 4, self.i * 4, kernel_size=4, padding=0)

        # field12
        self.conv5a = nn.Conv1d(self.i, self.i * 2, kernel_size=1, padding=0)
        self.conv5b = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=3, padding=0)
        self.conv5c = nn.Conv1d(self.i * 2, self.i * 4, kernel_size=5, padding=0)
        self.conv5d = nn.Conv1d(self.i * 4, self.i * 4, kernel_size=6, padding=0)

        # field 16
        self.conv6a = nn.Conv1d(self.i, self.i * 2, kernel_size=1, padding=0)
        self.conv6b = nn.Conv1d(self.i * 2, self.i * 2, kernel_size=6, padding=0)
        self.conv6c = nn.Conv1d(self.i * 2, self.i * 4, kernel_size=6, padding=0)
        self.conv6d = nn.Conv1d(self.i * 4, self.i * 4, kernel_size=6, padding=0)

        self.attention2 = SelfAttention(self.i * 6 * 2)
        self.convcomp = nn.Conv1d(self.i * 12 * 2, self.i * 6 * 2, kernel_size=1, padding=0)

        # self.ln1 = nn.LayerNorm(self.e_size * 2 * 5)  # Change size as needed

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = LSTMWithAttention(self.i * 6 * 2, self.i * 6 * 2)

        # self.attention3 = SelfAttention(self.e_size * 4 + (self.e_size * 2))

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_comp = nn.Linear(self.i * 6 * 2 * num_directions, self.i * 3 * 2 * num_directions)
        self.fc_note = nn.Linear(self.i * 3 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.i * 3 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.i * 3 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.i * 3 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = self.fc_embedding(embeddings)
        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)

        # embeddings = self.attention1(embeddings)

        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1a(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.relu(conv_out1)
        conv_out1 = self.conv1b(conv_out1)  # no padding required for kernel_size=1

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2a(padded_embeddings2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2b(conv_out2)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2c(conv_out2)
        conv_out2 = self.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3a(padded_embeddings3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3b(conv_out3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3c(conv_out3)
        conv_out3 = self.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4a(padded_embeddings4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4c(conv_out4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4d(conv_out4)
        conv_out4 = self.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (5, 6))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5a(padded_embeddings5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5b(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5c(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5d(conv_out5)
        conv_out5 = self.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings, (7, 8))  # Add 4 padding to left and 4 padding to right for kernel_size=9
        conv_out6 = self.conv6a(padded_embeddings6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6b(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6c(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6d(conv_out6)
        conv_out6 = self.relu(conv_out6)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention2(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc_comp(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnImpv3(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnImpv3, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches
        # field 1
        self.conv1 = nn.Conv1d(self.e_size, note_data.n_vocab, kernel_size=1, padding=0)
        self.conv11 = nn.Conv1d(note_data.n_vocab, self.e_size * 2, kernel_size=1, padding=0)

        # field 3
        self.conv2 = nn.Conv1d(note_data.n_vocab, self.e_size * 2, kernel_size=3)

        # field 7

        # field 4
        self.conv3 = nn.Conv1d(note_data.n_vocab, self.e_size * 2, kernel_size=4)

        # field 8
        self.conv4a = nn.Conv1d(note_data.n_vocab, self.e_size, kernel_size=2, padding=0)
        self.conv4b = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=4, padding=0)
        self.conv4c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)

        # field 16
        self.conv5 = nn.Conv1d(note_data.n_vocab, self.e_size * 2, kernel_size=16, padding=0)

        self.attention2 = SelfAttention(self.e_size * 5 * 2)

        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(note_data.n_vocab)

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 5 * 2, self.e_size * 5 * 2, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 5 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)

        # embeddings = self.attention1(embeddings)

        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = self.bn(conv_out1)
        conv_out1 = self.relu(conv_out1)  # no padding required for kernel_size=1

        conv_out11 = self.conv11(conv_out1)
        conv_out11 = self.relu(conv_out11)

        conv_out2 = F.pad(conv_out1, (1, 1))
        conv_out2 = self.conv2(conv_out2)
        conv_out2 = self.relu(conv_out2)

        conv_out3 = F.pad(conv_out1, (1, 2))
        conv_out3 = self.conv3(conv_out3)
        conv_out3 = self.relu(conv_out3)

        conv_out4 = F.pad(conv_out1, (3, 4))
        conv_out4 = self.conv4a(conv_out4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4b(conv_out4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4c(conv_out4)
        conv_out4 = self.relu(conv_out4)

        conv_out5 = F.pad(conv_out1, (7, 8))
        conv_out5 = self.conv5(conv_out5)
        conv_out5 = self.relu(conv_out5)

        conv_out = torch.cat((conv_out11, conv_out2, conv_out3, conv_out4, conv_out5), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention2(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class EmbConvLstmAttnImpv4(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstmAttnImpv4, self).__init__()

        self.embedding_size1 = int(math.sqrt(note_data.n_vocab))
        self.embedding_size2 = int(math.sqrt(note_data.o_vocab))
        self.embedding_size3 = int(math.sqrt(note_data.d_vocab))
        self.embedding_size4 = int(math.sqrt(note_data.v_vocab))
        self.bidirectional = bidirectional
        self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4

        # self.attention1 = SelfAttentionNorm(self.e_size)
        # Define two convolution branches
        # field 1
        self.conv1 = nn.Conv1d(self.e_size, self.e_size * 2, kernel_size=1, padding=0)

        # field 3

        self.conv2b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)
        self.conv2c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=1, padding=0)

        # field 4

        self.conv3b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=2, padding=0)
        self.conv3c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)

        # field 8

        self.conv4b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=2, padding=0)
        self.conv4c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        self.conv4d = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)

        # field12

        self.conv5b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=3, padding=0)
        self.conv5c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=5, padding=0)
        self.conv5d = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=6, padding=0)

        # field 16

        self.conv6b = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=2, padding=0)
        self.conv6c = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=2, padding=0)
        self.conv6d = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        self.conv6e = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=4, padding=0)
        self.conv6f = nn.Conv1d(self.e_size * 2, self.e_size * 2, kernel_size=8, padding=0)

        self.attention2 = SelfAttention(self.e_size * 6 * 2)

        # self.ln1 = nn.LayerNorm(self.e_size * 2 * 5)  # Change size as needed

        self.relu = nn.LeakyReLU()

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.e_size * 6 * 2, self.e_size * 6 * 2, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        # self.attention3 = SelfAttention(self.e_size * 4 + (self.e_size * 2))

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size * 6 * 2 * num_directions, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        # pos_enc = positional_encoding(seq_length, self.e_size, x.device)
        # embeddings += pos_enc.unsqueeze(0)

        # embeddings = self.attention1(embeddings)

        embeddings = embeddings.permute(0, 2, 1)

        embeddings = self.dropout(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        conv_out2 = self.conv2b(conv_out1)
        conv_out2 = self.relu(conv_out2)
        conv_out2 = self.conv2c(conv_out2)
        conv_out2 = self.relu(conv_out2)

        conv_out3 = self.relu(conv_out1)
        conv_out3 = F.pad(conv_out3, (1, 2))
        conv_out3 = self.conv3b(conv_out3)
        conv_out3 = self.relu(conv_out3)
        conv_out3 = self.conv3c(conv_out3)
        conv_out3 = self.relu(conv_out3)

        conv_out4 = self.conv4b(conv_out1)
        conv_out4 = F.pad(conv_out4, (3, 4))
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4c(conv_out4)
        conv_out4 = self.relu(conv_out4)
        conv_out4 = self.conv4d(conv_out4)
        conv_out4 = self.relu(conv_out4)

        conv_out5 = self.conv5b(conv_out1)
        conv_out5 = F.pad(conv_out5, (5, 6))
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5c(conv_out5)
        conv_out5 = self.relu(conv_out5)
        conv_out5 = self.conv5d(conv_out5)
        conv_out5 = self.relu(conv_out5)

        conv_out6 = self.conv6b(conv_out1)
        conv_out6 = F.pad(conv_out6, (7, 8))
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6c(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6d(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6e(conv_out6)
        conv_out6 = self.relu(conv_out6)
        conv_out6 = self.conv6f(conv_out6)
        conv_out6 = self.relu(conv_out6)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention2(conv_out)
        conv_out = self.dropout(conv_out)

        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class Mod34(nn.Module):
    def __init__(self, chan_in, chan_out, dropout_rate=0.5):
        super(Mod34, self).__init__()
        self.conv1 = nn.Conv1d(chan_in, chan_out, kernel_size=1, padding=0)

        self.conv31 = nn.Conv1d(chan_in, chan_out, kernel_size=1, padding=0)
        self.conv32 = nn.Conv1d(chan_out, chan_out, kernel_size=3, padding=1)
        self.conv41 = nn.Conv1d(chan_in, chan_out, kernel_size=1, padding=0)
        self.conv42 = nn.Conv1d(chan_out, chan_out, kernel_size=4, padding=2)
        self.conv42 = nn.Conv1d(chan_out, chan_out, kernel_size=4, padding=2)
        self.relu = nn.ReLU()

        # dimensionality reduction
        # self.conv4 = nn.Conv1d(num_channels * 4, num_channels, kernel_size=1)

    def forward(self, x):
        conv_out1 = self.conv1(x)
        conv_out1 = F.relu(conv_out1)

        conv_out3 = self.conv32(x)
        conv_out3 = F.relu(conv_out3)

        conv_out4 = self.conv42(x)
        conv_out4 = F.relu(conv_out4)

        cat = torch.cat((conv_out1, conv_out3, conv_out4), dim=1)
        return self.relu(cat)


class Incept(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(Incept, self).__init__()
        self.embedding_size1 = 96
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32
        self.bidirectional = bidirectional
        # self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.v = self.embedding_size1

        self.r1 = nn.Conv1d(self.e_size, self.v, kernel_size=1)
        self.conv1 = Mod34(self.v, self.v * 2, dropout)
        self.r2 = nn.Conv1d(self.v * 6, self.v * 2, kernel_size=1)
        self.conv2 = Mod34(self.v * 2, self.v * 2, dropout)
        self.r3 = nn.Conv1d(self.v * 6, self.v * 2, kernel_size=1)
        self.conv3 = Mod34(self.v * 2, self.v * 2, dropout)
        self.r4 = nn.Conv1d(self.v * 6, self.v * 2, kernel_size=1)
        self.conv4 = Mod34(self.v * 2, self.v * 2, dropout)
        self.r5 = nn.Conv1d(self.v * 6, self.v * 2, kernel_size=1)
        self.conv5 = Mod34(self.v * 2, self.v * 2, dropout)

        self.sconv11 = nn.Conv1d(self.e_size, self.v, kernel_size=1)
        self.sconv12 = nn.Conv1d(self.v, self.v * 2, kernel_size=3)

        self.sconv21 = nn.Conv1d(self.e_size, self.v, kernel_size=1)
        # self.sconv22 = nn.Conv1d(self.v, self.v * 2, kernel_size=4)
        self.sconv22 = nn.Conv1d(self.v, self.v * 2, kernel_size=4)

        # self.sconv31 = nn.Conv1d(self.e_size, self.v, kernel_size=1)
        # self.sconv32 = nn.Conv1d(self.v, self.v * 2, kernel_size=8)
        # self.rd  = nn.Conv1d(self.v * 12, , kernel_size=1)

        self.attention1 = SelfAttention(self.v * 6)
        self.attention2 = SelfAttention(self.v * 6)

        # Adjust the input size of LSTM layer to take  account concatenated output from conv layers
        self.lstm = nn.LSTM(self.v * 6, self.v * 6, num_layers=2, batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        num_directions = 2 if bidirectional else 1
        self.fc_note = nn.Linear(self.v * 6 * num_directions, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.v * 6 * num_directions, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.v * 6 * num_directions, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.v * 6 * num_directions, note_data.v_vocab)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        conv_out_l = self.r1(embeddings)
        conv_out_l = self.conv1(conv_out_l)
        conv_out_l = self.r2(conv_out_l)
        conv_out_l = self.conv2(conv_out_l)
        conv_out_l = self.r3(conv_out_l)
        conv_out_l = self.conv3(conv_out_l)
        conv_out_l = self.r4(conv_out_l)
        conv_out_l = self.conv4(conv_out_l)
        conv_out_l = self.r5(conv_out_l)
        conv_out_l = self.conv5(conv_out_l)

        conv_out_s1 = self.sconv11(F.pad(embeddings, (1, 1)))
        conv_out_s1 = self.sconv12(F.relu(conv_out_s1))
        conv_out_s1 = F.relu(conv_out_s1)

        conv_out_s2 = self.sconv21(F.pad(embeddings, (1, 2)))
        conv_out_s2 = self.sconv22(F.relu(conv_out_s2))
        conv_out_s2 = F.relu(conv_out_s2)

        conv_out_s3 = self.sconv31(F.pad(embeddings, (3, 4)))
        conv_out_s3 = self.sconv32(F.relu(conv_out_s3))
        conv_out_s3 = F.relu(conv_out_s3)

        conv_out1 = torch.cat((conv_out_s1, conv_out_s2, conv_out_s3), dim=1)
        conv_out1 = self.attention1(conv_out1.permute(0, 2, 1))

        conv_out2 = self.attention2(conv_out_l.permute((0, 2, 1)))

        conv_out = (conv_out1 + conv_out2) / 2

        conv_out = F.relu(conv_out)
        conv_out = self.dropout(conv_out)
        lstm_out, _ = self.lstm(conv_out)

        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, : self.e_size * 2], lstm_out[:, 0, self.e_size * 2:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class LSTM_ARCH(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_dim, num_layers=None, lstm=None, dropout_rate=0.2, fc=True):
        super(LSTM_ARCH, self).__init__()
        self.fc = fc
        self.ln2 = nn.LayerNorm(input_size)
        self.hidden_dim = hidden_dim

        if lstm is None:
            self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        else:
            self.lstm = lstm
        self.attention = SelfAttentionNorm(hidden_dim)
        if self.fc:
            self.fc = nn.Linear(hidden_dim, vocab_size)  # output layer
        self.comp = ExComp(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(2, batch_size, self.hidden_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        a_out = self.attention(out)
        out = torch.add(out, a_out)
        out = self.ln2(out)
        out_b = self.comp(out)
        out = torch.mul(out, out_b)

        # Ensure that out has three dimensions before indexing
        if out.dim() == 2:
            out = out.unsqueeze(1)

        # out = out[:, -1, :]
        out = self.dropout(out)
        if self.fc:
            out = self.fc(out)  # apply the output layer to the LSTM's outputs
        return out, hidden


class ExComp(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ExComp, self).__init__()
        self.up = nn.Linear(input_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.down(F.relu(self.up(x)))


class SelfAttentionNorm(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionNorm, self).__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.ln(self.value(x))
        # weights = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1]), dim=-1)
        weights = F.softmax(q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(k.shape[-1])), dim=-1)
        return weights @ v


class EmbConvLstmNew(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstmNew, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64
        self.bidirectional = bidirectional
        # self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0)

        self.incpt = nn.Sequential(
            Mod34
        )

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
        #                     bidirectional=bidirectional)
        # self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size* 8, 2, True)
        self.lstm = nn.LSTM(self.comp_size * 10, self.comp_size * 10, 2, batch_first=True)

        self.fc_out = nn.Linear(self.comp_size * 10, self.e_size)
        self.dropout = nn.Dropout(dropout)

        self.fc_note = nn.Linear(self.e_size, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size, note_data.v_vocab)
        # self.fc_note = nn.Linear(self.comp_size * 10, note_data.n_vocab)
        # self.fc_offset = nn.Linear( self.comp_size * 10, note_data.o_vocab)
        # self.fc_duration = nn.Linear( self.comp_size * 10, note_data.d_vocab)
        # self.fc_velocity = nn.Linear( self.comp_size * 10 , note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = self.in_linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        # incept_out = self.incpt(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.relu(conv_out5)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5), dim=1)
        # conv_out = torch.add(conv_out, incept_out)
        conv_out = F.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)

        lstm_out = self.lstm(conv_out)

        lstm_out = lstm_out[:, -1]

        lstm_out = self.fc_out(lstm_out)

        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


# class PolyCNNLSTM(nn.Module):
#     def __init__(self, note_data,num_chan, dropout=0.3):
#         super(PolyCNNLSTM, self).__init__()
#         pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))
#         self.note_data = note_data
#         self.num_chan = num_chan
#
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(note_data.n_vocab, 64),
#             nn.Embedding(note_data.o_vocab, 64),
#             nn.Embedding(note_data.d_vocab, 64),
#             nn.Embedding(note_data.v_vocab, 64)]
#         )
#
#         self.e_size = 256 * num_chan
#         self.comp_size = pow2(self.e_size / num_chan)
#         self.fc_encode = nn.Linear(self.e_size, self.comp_size)
#         self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0)
#         self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0)
#         self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0)
#         self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0)
#
#
#         self.incpt = nn.Sequential (
#             IncepModule(self.comp_size, self.comp_size),
#             IncepModule( self.comp_size * 4, self.comp_size),
#             IncepModule(self.comp_size * 4, self.comp_size),
#             IncepModule(self.comp_size * 4, self.comp_size),
#             IncepBottleNeckModule(self.comp_size * 4, self.comp_size * 2, self.comp_size  * 8)
#         )
#
#         # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
#         # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
#         #                     bidirectional=bidirectional)
#         self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size* 8, 2, True)
#        # self._lstm = nn.LSTM(self.comp_size * 8, self.comp_size* 8, 2, batch_first=True)
#         self.lstm = LSTM_ARCH(self.e_size, self.comp_size * 8, self.comp_size * 8 , lstm=self._lstm, dropout_rate=dropout)
#
#         self.dropout = nn.Dropout(dropout)
#         print(self.e_size)
#         self.fc_decode = nn.Linear(self.comp_size * 8, self.e_size)
#         self.fc_note = nn.Linear(self.e_size , note_data.n_vocab)
#         self.fc_offset = nn.Linear(self.e_size , note_data.o_vocab)
#         self.fc_duration = nn.Linear(self.e_size, note_data.d_vocab)
#         self.fc_velocity = nn.Linear(self.e_size , note_data.v_vocab)
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#         batch_size, seq_length, num_tuples, num_features = x.size()
#         print(len(self.embeddings))
#         print(num_features)
#         print(x.size())
#         # Assumes that the input x has dimensions (batch_size, seq_length, num_tuples, num_features)
#
#         # Embedding for each feature
#        # embeddings = [self.embeddings[i](x[..., i].long()) for i in range(num_features)]
#         batch_size, seq_length, num_tuples, num_features = x.size()
#
#         embeddings = []
#         for i in range(num_tuples):
#             tuple_embeddings = [self.embeddings[j](x[..., i, j].long()) for j in range(num_features)]
#             tuple_embeddings = torch.cat(tuple_embeddings, dim=-1)
#             embeddings.append(tuple_embeddings)
#
#
#         # embeddings = torch.stack(embeddings, dim=2)
#         #
#         #
#         #
#         # embeddings = embeddings.view(embeddings.size(0), embeddings.size(1), -1)
#         # embeddings = F.relu(embeddings)
#         # print(embeddings.size())
#         # embeddings = self.fc_encode(embeddings)
#         # embeddings = embeddings.permute(0, 2, 1)
#         # embeddings = self.dropout(embeddings)
#
#         embeddings = torch.stack(embeddings, dim=2)
#         embeddings = embeddings.view(batch_size, seq_length, -1)
#         embeddings = F.relu(embeddings)
#
#         # Reshape embeddings before passing to fully connected layer
#         embeddings = embeddings.view(-1, self.e_size)
#         embeddings = self.fc_encode(embeddings)
#       #  embeddings = embeddings.view(batch_size, seq_length, num_tuples, -1)
#         embeddings = embeddings.view(batch_size, num_tuples * 64, seq_length)
#
#         # embeddings = embeddings.permute(0, 2, 1)
#         embeddings = self.dropout(embeddings)
#
#         incept_out = self.incpt(embeddings)
#
#         # Apply the convolutions
#         conv_out1 = F.relu(self.conv1(embeddings))
#         conv_out2 = F.relu(self.conv2(F.pad(embeddings, (1, 1))))
#         conv_out3 = F.relu(self.conv3(F.pad(embeddings, (1, 2))))
#         conv_out4 = F.relu(self.conv4(F.pad(embeddings, (3, 4))))
#
#         # Concatenate the outputs along the channel dimension
#         conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
#         conv_out = torch.mul(conv_out, incept_out)
#         conv_out = F.relu(conv_out)
#         conv_out = self.dropout(conv_out)
#
#         conv_out = conv_out.permute(0, 2, 1)
#
#         lstm_out = self.lstm(conv_out)
#
#         lstm_out = lstm_out.contiguous().view(-1, self.comp_size * 8)
#         lstm_out = self.fc_decode(lstm_out)  # this should reshape
#
#         # Apply the fully connected layers
#         notes = self.fc_note(lstm_out)
#         offsets = self.fc_offset(lstm_out)
#         durations = self.fc_duration(lstm_out)
#         velocities = self.fc_velocity(lstm_out)
#
#         print(notes.size())
#         return notes, offsets, durations, velocities


class PolyCNNLSTM(nn.Module):
    def __init__(self, note_data, num_chan=8, dropout=0.3, bidirectional=False):
        super(PolyCNNLSTM, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))
        self.note_data = note_data
        self.num_chan = num_chan

        self.embeddings = nn.ModuleList([
            nn.Embedding(note_data.n_vocab, 16),
            nn.Embedding(note_data.o_vocab, 16),
            nn.Embedding(note_data.d_vocab, 16),
            nn.Embedding(note_data.v_vocab, 16)]
        )

        self.e_size = 64 * num_chan
        self.comp_size = pow2(self.e_size / num_chan)
        self.fc_encode = nn.Linear(self.e_size, self.comp_size)
        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0)

        self.incpt = nn.Sequential(
            IncepModule(self.comp_size, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepBottleNeckModule(self.comp_size * 4, self.comp_size * 2, self.comp_size * 8)
        )

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
        #                     bidirectional=bidirectional)
        self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size * 8, 2, True)
        # self._lstm = nn.LSTM(self.comp_size * 8, self.comp_size* 8, 2, batch_first=True)
        self.lstm = LSTM_ARCH(self.e_size, self.comp_size * 8, self.comp_size * 8, lstm=self._lstm,
                              dropout_rate=dropout)

        self.dropout = nn.Dropout(dropout)
        print(self.e_size)
        self.fc_decode = nn.Linear(self.comp_size * 8, self.e_size)
        self.fc_note = nn.Linear(self.e_size, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size, note_data.v_vocab)

        self.fc_out = nn.Linear(self.comp_size * 8, 4)

        self._initialize_weights()

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.e_size * 10).to(device)
        cell = torch.zeros(2, batch_size, self.e_size * 10).to(device)
        return hidden, cell

    def forward(self, x):
        batch_size, seq_length, num_tuples, num_features = x.size()
        print(len(self.embeddings))
        print(num_features)
        print(x.size())
        # Assumes that the input x has dimensions (batch_size, seq_length, num_tuples, num_features)

        # Embedding for each feature
        # embeddings = [self.embeddings[i](x[..., i].long()) for i in range(num_features)]
        batch_size, seq_length, num_tuples, num_features = x.size()

        embeddings = []
        for i in range(num_tuples):
            tuple_embeddings = [self.embeddings[j](x[..., i, j].long()) for j in range(num_features)]
            tuple_embeddings = torch.cat(tuple_embeddings, dim=-1)
            embeddings.append(tuple_embeddings)

        # embeddings = torch.stack(embeddings, dim=2)
        #
        #
        #
        # embeddings = embeddings.view(embeddings.size(0), embeddings.size(1), -1)
        # embeddings = F.relu(embeddings)
        # print(embeddings.size())
        # embeddings = self.fc_encode(embeddings)
        # embeddings = embeddings.permute(0, 2, 1)
        # embeddings = self.dropout(embeddings)

        embeddings = torch.stack(embeddings, dim=2)
        embeddings = embeddings.view(batch_size, seq_length, -1)
        embeddings = F.relu(embeddings)

        # Reshape embeddings before passing to fully connected layer
        embeddings = embeddings.view(-1, self.e_size)
        embeddings = self.fc_encode(embeddings)
        #  embeddings = embeddings.view(batch_size, seq_length, num_tuples, -1)
        embeddings = embeddings.view(batch_size, num_tuples * 64, seq_length)

        # embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        incept_out = self.incpt(embeddings)

        # Apply the convolutions
        conv_out1 = F.relu(self.conv1(embeddings))
        conv_out2 = F.relu(self.conv2(F.pad(embeddings, (1, 1))))
        conv_out3 = F.relu(self.conv3(F.pad(embeddings, (1, 2))))
        conv_out4 = F.relu(self.conv4(F.pad(embeddings, (3, 4))))

        # Concatenate the outputs along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = torch.mul(conv_out, incept_out)
        conv_out = F.relu(conv_out)
        conv_out = self.dropout(conv_out)

        conv_out = conv_out.permute(0, 2, 1)

        lstm_out = self.lstm(conv_out)
        print(lstm_out.shape)

        lstm_out = lstm_out.contiguous().view(-1, self.comp_size * 4)  # print(lstm_out.shape)
        # split the lstm_out tensor along the last dimension
        outputs = torch.split(lstm_out, 1, dim=-1)

        # remove the last dimension of size 1
        outputs = [out.squeeze(-1) for out in outputs]

        return outputs  # returns a list of 4 tensors

    #   lstm_out = self.lstm(conv_out)
    #
    #   lstm_out = lstm_out.contiguous().view(-1, self.comp_size * 8)
    # #  lstm_out = self.fc_decode(lstm_out)  # this should reshape
    #
    #   # output = []
    #   # for i in range(num_tuples):
    #   #     notes = self.fc_note[i](lstm_out)
    #   #     offsets = self.fc_offset[i](lstm_out)
    #   #     durations = self.fc_duration[i](lstm_out)
    #   #     velocities = self.fc_velocity[i](lstm_out)
    #   #
    #   #     output.append((notes, offsets, durations, velocities))
    #   #
    #   # print(output)
    #
    #   outputs = self.fc_out(lstm_out)
    #
    #   return outputs  # return list of tuples

    # # Apply the fully connected layers
    # notes = self.fc_note(lstm_out)
    # offsets = self.fc_offset(lstm_out)
    # durations = self.fc_duration(lstm_out)
    # velocities = self.fc_velocity(lstm_out)
    #
    # print(notes.size())
    # return notes, offsets, durations, velocities


class EmbConvLstmNew2(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstmNew2, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 32
        self.embedding_size2 = 96
        self.embedding_size3 = 96
        self.embedding_size4 = 32
        self.bidirectional = bidirectional
        # self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)
        self.e_size = 256
        self.comp_size = pow2(self.e_size / 2)
        self.in_linear = nn.Linear(self.e_size, self.comp_size)

        # self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0)
        # self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0)
        # self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0)
        # self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0)
        # self.relu = nn.ReLU()

        self.incpt1 = nn.Sequential(
            IncepModule(self.comp_size, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
        )

        self.incpt2 = nn.Sequential(
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
        )

        self.midatt = SelfAttentionNorm(self.comp_size)

        self.incpt3 = nn.Sequential(
            IncepModule(self.comp_size * 9, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepBottleNeckModule(self.comp_size * 4, self.comp_size * 4, self.comp_size * 8)
        )

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
        #                     bidirectional=bidirectional)
        self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size * 8, 2, True)
        # self._lstm = nn.LSTM(self.comp_size * 8, self.comp_size* 8, 2, batch_first=True)
        self.lstm = LSTM_ARCH(self.e_size, self.comp_size * 8, self.comp_size * 8, lstm=self._lstm,
                              dropout_rate=dropout)

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)
        self.dropout = nn.Dropout(dropout)

        self.fc_note = nn.Linear(self.e_size, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()
        #
        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = self.in_linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        incept_out = self.incpt1(embeddings)
        incept_out = self.incpt2(F.relu(incept_out))
        incept_out = F.relu((incept_out))
        incept_attn = self.midatt(incept_out)

        conv_out = torch.cat((embeddings, incept_attn, incept_out), dim=1)
        conv_out = self.incpt3(conv_out)

        conv_out = F.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        lstm_out = self.lstm(conv_out)
        #   lstm_out = lstm_out[:, -1]

        # lstm_out = self.fc_out(lstm_out)
        lstm_out = self.dropout(lstm_out)
        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        y = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y


class PolyphonicLSTM(nn.Module):
    def __init__(self, note_data):
        super(PolyphonicLSTM, self).__init__()

        # Define the embedding layers
        self.note_data = note_data
        self.note_embedding = nn.Embedding(note_data.n_vocab, 64)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, 64)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, 64)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, 64)

        # Define the LSTM
        self.lstm = nn.LSTM(input_size=4 * 64 * 4,
                            hidden_size=512,
                            num_layers=2,
                            batch_first=True)

        # Define the output layers
        self.out_note = TimeDistributed(nn.Linear(512, note_data.n_vocab * 4))
        self.out_offset = TimeDistributed(nn.Linear(512, note_data.o_vocab * 4))
        self.out_duration = TimeDistributed(nn.Linear(512, note_data.d_vocab * 4))
        self.out_velocity = TimeDistributed(nn.Linear(512, note_data.v_vocab * 4))

    def forward(self, x):
        # Extract the separate fields
        notes = x[..., 0]
        offsets = x[..., 1]
        durations = x[..., 2]
        velocities = x[..., 3]

        # Pass through the embedding layers
        note_embed = self.note_embedding(notes.long())
        offset_embed = self.offset_embedding(offsets.long())
        duration_embed = self.duration_embedding(durations.long())
        velocity_embed = self.velocity_embedding(velocities.long())

        # Concatenate the embeddings
        x = torch.cat((note_embed, offset_embed, duration_embed, velocity_embed), -1)
        x = x.view(x.size(0), x.size(1), -1)  # flatten the last two dimensions

        # Pass through LSTM
        out, _ = self.lstm(x)
        print("out before reshape:", out.shape)
        out = out[:, -1, :]
        print("out after reshape:", out.shape)
        # Pass through output layers
        out_note = self.out_note(out)  # out_note has the shape [batch_size, vocab_size]
        out_offset = self.out_offset(out)  # out_offset has the shape [batch_size, vocab_size]
        out_duration = self.out_duration(out)  # out_duration has the shape [batch_size, vocab_size]
        out_velocity = self.out_velocity(out)  # out_velocity has the shape [batch_size, vocab_size]

        out_note = out_note.view(out.size(0), 4, -1)
        out_offset = out_offset.view(out.size(0), 4, -1)
        out_duration = out_duration.view(out.size(0), 4, -1)
        out_velocity = out_velocity.view(out.size(0), 4, -1)

        print(out_offset.shape)
        # Combine the outputs
        out = (out_note, out_offset, out_duration, out_velocity)

        return out


# class MidiLSTM(nn.Module):
#     def __init__(self, note_data):
#         super(MidiLSTM, self).__init__()
#
#         self.note_embedd = nn.Embedding(note_data.n_vocab, 64)
#         self.offset_embedd = nn.Embedding(note_data.o_vocab, 64)
#         self.duration_embedd = nn.Embedding(note_data.d_vocab, 64)
#         self.velocity_embedd = nn.Embedding(note_data.v_vocab, 64)
#
#         self.lstm = nn.LSTM(256 * 6, 256 * 6, batch_first=True)
#
#         self.note_out = nn.Linear(256 * 6, note_data.n_vocab)
#         self.offset_out = nn.Linear(256 * 6, note_data.o_vocab)
#         self.duration_out = nn.Linear(256 * 6, note_data.d_vocab)
#         self.velocity_out = nn.Linear(256 * 6, note_data.v_vocab)
#
#     def forward(self, x):
#
#         notes = x[:, :, 0, :]  # Shape should be [batch_size, sequence_length, length_feature_array]
#         offsets = x[:, :, 1, :]
#         durations = x[:, :, 2, :]
#         velocities = x[:, :, 3, :]
#         note_embedd = self.note_embedd(notes.long())
#         offset_embedd = self.offset_embedd(offsets.long())
#         duration_embedd = self.duration_embedd(durations.long())
#         velocity_embedd = self.velocity_embedd(velocities.long())
#
#         embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
#         embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
#
#
#         lstm_out, _ = self.lstm( embeddings)
#         lstm_out = lstm_out[:, -1, :]
#         note_pred = self.note_out(lstm_out)
#         offset_pred = self.offset_out(lstm_out)
#         duration_pred = self.duration_out(lstm_out)
#         velocity_pred = self.velocity_out(lstm_out)
#         print(note_pred.shape)
#       return note_pred, offset_pred, duration_pred, velocity_pred

class MidiLSTM(nn.Module):
    def __init__(self, note_data):
        super(MidiLSTM, self).__init__()
        self.note_data = note_data
        self.note_embedd = nn.Embedding(note_data.n_vocab, 64)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, 64)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, 64)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, 64)

        self.conv = nn.Sequential(
            nn.Conv1d(1152, 192, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(192, 384, kernel_size=8, padding=0),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(256 * 6, 256 * 6, 2, batch_first=True)

        self.note_out = nn.Linear(256 * 6, note_data.n_vocab * 6)  # Adjust output size
        self.offset_out = nn.Linear(256 * 6, note_data.o_vocab * 6)  # Adjust output size
        self.duration_out = nn.Linear(256 * 6, note_data.d_vocab * 6)  # Adjust output size
        self.velocity_out = nn.Linear(256 * 6, note_data.v_vocab * 6)  # Adjust output size

    def forward(self, x):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]
        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        embeddings = embeddings.permute(0, 2, 1)
        conv_out = self.conv(embeddings)
        conv_out = conv_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.note_out(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.offset_out(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.duration_out(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.velocity_out(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU()  # use sigmoid if input is normalized between [0,1], else use ReLU or similar
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PolyCNNLSM(nn.Module):
    def __init__(self, note_data, dropout=0.4):
        super(PolyCNNLSM, self).__init__()
        self.note_data = note_data
        self.note_embedd = nn.Embedding(note_data.n_vocab, 32)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, 64)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, 64)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, 32)
        self.dropout = dropout

        self.conv1 = nn.Conv1d(1152, 384, kernel_size=1, padding=0)

        self.conv2 = nn.Sequential(
            nn.Conv1d(1152, 192, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(192, 384, kernel_size=3, padding=0),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(1152, 192, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(192, 384, kernel_size=4, padding=0),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(1152, 192, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(192, 384, kernel_size=8, padding=0),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(1152, 192, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(192, 384, kernel_size=16, padding=0),
            nn.ReLU()
        )

        self.attention = SelfAttentionImpr(640)

        self.convcomp = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(1920, 640, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(640, 1280, 2, batch_first=True, dropout=dropout)

        self.note_out = nn.Linear(1280, note_data.n_vocab * 6)  # Adjust output size
        self.offset_out = nn.Linear(1280, note_data.o_vocab * 6)  # Adjust output size
        self.duration_out = nn.Linear(1280, note_data.d_vocab * 6)  # Adjust output size
        self.velocity_out = nn.Linear(1280, note_data.v_vocab * 6)  # Adjust output size

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]
        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        embeddings = F.dropout(embeddings, self.dropout)
        embeddings = embeddings.permute(0, 2, 1)

        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)

        padded_embeddings5 = F.pad(embeddings, (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5), dim=1)
        conv_out = F.dropout(conv_out, self.dropout)

        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.attention(conv_out)
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.note_out(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.offset_out(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.duration_out(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.velocity_out(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred


class PolyCNNLSM2(nn.Module):
    def __init__(self, note_data, dropout=0.25):
        super(PolyCNNLSM2, self).__init__()
        self.note_data = note_data
        self.note_embedd = nn.Embedding(note_data.n_vocab, 16)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, 48)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, 48)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, 16)
        self.dropput = dropout

        self.encode1 = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU()
        )

        self.conv1 = nn.Conv1d(192, 384, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(192, 384, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(192, 384, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(192, 384, kernel_size=8, padding=0)
        self.conv5 = nn.Conv1d(192, 384, kernel_size=16, padding=0)

        self.encode2 = nn.Sequential(
            nn.Linear(1920, 960),
            nn.ReLU(),
        )

        self.attention = SelfAttentionNorm(960)

        self.lstm = nn.LSTM(960, 1920, 2, batch_first=True, dropout=dropout)

        self.encode3 = nn.Sequential(
            nn.Linear(1920, 960),
            nn.ReLU(),
        )

        self.note_out = nn.Linear(960, note_data.n_vocab * 6)  # Adjust output size
        self.offset_out = nn.Linear(960, note_data.o_vocab * 6)  # Adjust output size
        self.duration_out = nn.Linear(960, note_data.d_vocab * 6)  # Adjust output size
        self.velocity_out = nn.Linear(960, note_data.v_vocab * 6)  # Adjust output size

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]
        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        torch.Size([256, 64, 960])
        embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        embeddings = F.dropout(embeddings, self.dropput)
        embeddings = self.encode1(embeddings)
        embeddings = embeddings.permute(0, 2, 1)

        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.dropout(F.relu(conv_out1), self.dropput)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.dropout(F.relu(conv_out2), self.dropput)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.dropout(F.relu(conv_out3), self.dropput)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.dropout(F.relu(conv_out4), self.dropput)

        padded_embeddings5 = F.pad(embeddings, (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.dropout(F.relu(conv_out5), self.dropput)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5), dim=1)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.encode2(conv_out)
        conv_out = self.attention(conv_out)
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.encode3(lstm_out)
        print("lstm_out.shape:", lstm_out.shape)
        print("self.fc_note(lstm_out).shape:", self.note_out(lstm_out).shape)

        note_pred = self.note_out(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.offset_out(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.duration_out(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.velocity_out(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred


class EmbConvLstPoly(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32
        self.bidirectional = bidirectional
        # self.note_data = note_data

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.e_size = 128
        self.comp_size = 128
        self.comp = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=16, padding=0)
        self.relu = nn.ReLU()

        self.incpt = nn.Sequential(
            IncepModule(self.comp_size, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size * 2)
        )

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
        #                     bidirectional=bidirectional)
        # self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size* 8, 2, True)
        self._lstm = nn.LSTM(self.comp_size * 8, self.comp_size * 8, 2, batch_first=True)
        self.lstm = LSTM_ARCH(self.comp_size * 4, self.comp_size * 8, self.comp_size * 8, lstm=self._lstm,
                              dropout_rate=dropout)

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)
        self.dropout = nn.Dropout(dropout)

        self.fc_note = nn.Linear(self.comp_size * 4, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 4, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 4, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 4, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]
        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        torch.Size([256, 64, 960])
        embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        embeddings = self.comp(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        incept_out = self.incpt(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.relu(conv_out4)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = torch.mul(conv_out, incept_out)
        conv_out = F.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)

        lstm_out = self.lstm(conv_out)
        lstm_out = lstm_out[:, -1, :]
        # print(lstm_out.shape)

        # lstm_out = lstm_out[:, -1, :]
        # lstm_out = lstm_out[:, -1, :]
        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class EmbConvLstPoly2(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly2, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.e_size = 128
        self.comp_size = 256
        # self.comp = nn.Sequential(
        #     nn.Linear(1152, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 384),
        #     nn.ReLU(),
        #     nn.Linear(384, 192),
        #     nn.ReLU(),
        # )

        self.comp = nn.Sequential(
            nn.Conv1d(self.comp_size * 6, self.comp_size * 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv1d(self.comp_size * 3, int(self.comp_size * 1.5), 1, padding=0),
            nn.ReLU(),
        )
        self.complin = nn.Linear(int(self.comp_size * 1.5), self.comp_size)

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0)
        self.relu = nn.ReLU()

        self.incpt = nn.Sequential(
            IncepModule(self.comp_size, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size * 2),
            IncepModule(self.comp_size * 4 * 2, self.comp_size * 2),
            IncepBottleNeckModule(self.comp_size * 4 * 2, self.comp_size * 2, self.comp_size * 8)
        )

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
        #                     bidirectional=bidirectional)
        # self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size* 8, 2, True)

        # self.comp2att1 = SelfAttentionNorm(self.comp_size * 8)
        self.comp2 = nn.Sequential(
            nn.Conv1d(self.comp_size * 8, self.comp_size * 4, 1, padding=0),
            nn.ReLU(),
            nn.Conv1d(self.comp_size * 4, self.comp_size * 2, 1, padding=0),
            nn.ReLU(),
        )
        self.comp2att2 = SelfAttentionImpr(self.comp_size * 2)
        self.complin2 = nn.Linear(self.comp_size * 2, self.comp_size)

        self.lstm = nn.LSTM(self.comp_size, self.comp_size * 2, 2, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 6, self.comp_size , self.comp_size * 2 , lstm=self._lstm, dropout_rate=dropout)

        self.decodelin = nn.Linear(self.comp_size * 2, self.comp_size * 4)  # check input size

        self.decode = nn.Sequential(
            nn.ConvTranspose1d(self.comp_size * 4, self.comp_size * 5, 1, padding=0),  # Transposed convolutions
            nn.ReLU(),
            nn.ConvTranspose1d(self.comp_size * 5, self.comp_size * 6, 1, padding=0),  # Transposed convolutions
            nn.ReLU(),
        )

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)
        self.dropout = nn.Dropout(dropout)

        self.fc_note = nn.Linear(self.comp_size * 6, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 6, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 6, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 6, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]
        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)

        embeddings = self.dropout(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.comp(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.relu(self.complin(embeddings))
        embeddings = embeddings.permute(0, 2, 1)

        incept_out = self.incpt(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.relu(conv_out4)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = torch.mul(conv_out, incept_out)
        # conv_out = conv_out.permute(0, 2, 1)
        # conv_out = self.comp2att1(conv_out)
        #  conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.comp2(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.comp2att2(conv_out)
        conv_out = F.relu(self.complin2(conv_out))
        conv_out = self.dropout(conv_out)
        #  conv_out = conv_out.permute(0, 2, 1)

        lstm_out, _ = self.lstm(conv_out)

        # lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.decodelin(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.decode(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)

        lstm_out = lstm_out[:, -1, :]
        # print(lstm_out.shape)

        # lstm_out = lstm_out[:, -1, :]
        # lstm_out = lstm_out[:, -1, :]
        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred


class EmbConvLstPoly3(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly3, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.comp_size = 384

        # self.embeddattn = SelfAttentionNorm(1536)
        # self.comp = nn.Sequential(
        #     nn.Linear(1536, 768),
        #     nn.LayerNorm(768),
        #     nn.ReLU(),
        #     nn.Linear(768, 384),
        #     nn.LayerNorm(384),
        #     nn.ReLU(),
        #   #  nn.Linear(384, 256),
        #    # nn.LayerNorm(256),
        #    # nn.ReLU(),
        #     nn.Dropout(dropout),
        # )

        self.comp = nn.Sequential(
            nn.Conv1d(1536, 768, 1),
            # SelfAttentionNorm(768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Conv1d(768, 384, 1),
            nn.BatchNorm1d(384),
            nn.ReLU(),

        )

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0)
        #   self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=16, padding=0)
        self.relu = nn.ReLU()

        self.convattn = SelfAttentionNorm(self.comp_size * 4 * 1)

        self.convcomp = nn.Conv1d(self.comp_size * 8, self.comp_size * 4, 1)

        self.lstm = nn.LSTM(self.comp_size * 4 * 1, self.comp_size * 4 * 1, 2, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        # self.decode = nn.Sequential(
        #     nn.ConvTranspose1d(self.comp_size * 4, self.comp_size * 5, 1, padding=0),  # Transposed convolutions
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(self.comp_size * 5, self.comp_size * 6, 1, padding=0),  # Transposed convolutions
        #     nn.ReLU(),
        # )

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.fc_note = nn.Linear(self.comp_size * 4, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 4, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 4, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 4, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        return hidden, cell

    def forward(self, x):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]
        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        # embeddings = self.embeddattn(F.relu(embeddings))
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.comp(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.relu(conv_out4)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = F.relu(conv_out)
        conv_out = self.convattn(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)
        lstm_out, hidden = self.lstm(conv_out)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class EmbConvLstPoly4(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly4, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 50
        self.embedding_size2 = 50
        self.embedding_size3 = 50
        self.embedding_size4 = 50

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.comp_size = 512

        self.note_encode = EmbEncode(300, 128)
        self.offset_encode = EmbEncode(300, 128)
        self.duration_encode = EmbEncode(300, 128)
        self.velocity_encode = EmbEncode(300, 128)

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0)
        #   self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=16, padding=0)
        self.relu = nn.ReLU()

        # self.convattn = SelfAttentionNorm(self.comp_size * 4 * 1)

        self.convcomp = nn.Sequential(
            nn.Conv1d(self.comp_size * 8, self.comp_size * 4, 1),
            nn.Mish(),
            Permute(0, 2, 1),
            SelfAttentionNorm(self.comp_size * 4),
            Permute(0, 2, 1),
            nn.Conv1d(self.comp_size * 4, self.comp_size * 2, 1),
            nn.Mish(),
        )

        self.lstm = nn.LSTM(self.comp_size * 2 * 1, self.comp_size * 2 * 1, 2, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        # self.decode = nn.Sequential(
        #     nn.ConvTranspose1d(self.comp_size * 4, self.comp_size * 5, 1, padding=0),  # Transposed convolutions
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(self.comp_size * 5, self.comp_size * 6, 1, padding=0),  # Transposed convolutions
        #     nn.ReLU(),
        # )

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1).permute(0, 2, 1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1).permute(0, 2, 1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1).permute(0, 2,
                                                                                                                   1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1).permute(0,
                                                                                                                      2,
                                                                                                                      1)

        note_embedd = self.note_encode(note_embedd)
        offset_embedd = self.offset_encode(offset_embedd)
        duration_embedd = self.duration_encode(duration_embedd)
        velocity_embedd = self.velocity_encode(velocity_embedd)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=1)
        # embeddings = embeddings.view(embeddings.shape[0], -1, embeddings.shape[2])
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.mish(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.mish(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.mish(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.mish(conv_out4)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = F.mish(conv_out)
        #   conv_out = self.convattn(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)
        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly5(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly5, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.comp_size = 768

        # self.embeddattn = SelfAttentionNorm(1536)
        # self.comp = nn.Sequential(
        #     nn.Linear(1536, 768),
        #     nn.LayerNorm(768),
        #     nn.ReLU(),
        #     nn.Linear(768, 384),
        #     nn.LayerNorm(384),
        #     nn.ReLU(),
        #   #  nn.Linear(384, 256),
        #    # nn.LayerNorm(256),
        #    # nn.ReLU(),
        #     nn.Dropout(dropout),
        # )

        # self.comp = nn.Sequential(
        #     nn.Conv1d(1536, 768, 1),
        #    # SelfAttentionNorm(768),
        #     nn.BatchNorm1d(768),
        #     nn.ReLU(),
        #     nn.Conv1d(768, 384, 1),
        #     nn.BatchNorm1d(384),
        #     nn.ReLU(),
        #
        # )

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=8, padding=0)
        #   self.conv5 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=16, padding=0)

        # self.convattn = SelfAttentionNorm(self.comp_size * 4 * 1)
        #
        # self.convcomp = nn.Sequential(
        #     nn.Conv1d(self.comp_size * 8, self.comp_size * 4, 1),
        #     nn.Mish(),
        #     Permute(0,2,1),
        #     SelfAttentionNorm(self.comp_size * 4),
        #     Permute(0,2,1),
        #     nn.Conv1d(self.comp_size * 4, self.comp_size * 2, 1),
        #     nn.Mish(),
        # )

        self.lstm = nn.LSTM(self.comp_size * 4 * 1, self.comp_size * 1 * 4, 2, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        # self.decode = nn.Sequential(
        #     nn.ConvTranspose1d(self.comp_size * 4, self.comp_size * 5, 1, padding=0),  # Transposed convolutions
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(self.comp_size * 5, self.comp_size * 6, 1, padding=0),  # Transposed convolutions
        #     nn.ReLU(),
        # )

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.fc_note = nn.Linear(self.comp_size * 4, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 4, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 4, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 4, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]
        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        torch.Size([256, 64, 960])
        embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        embeddings = F.dropout(embeddings, self.dropout)
        embeddings = embeddings.permute(0, 2, 1)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.mish(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.mish(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.mish(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.mish(conv_out4)

        # padded_embeddings5 = F.pad(embeddings, (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        # conv_out5 = self.conv5(padded_embeddings5)
        # conv_out5 = F.mish(conv_out5)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)

        #   conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = F.mish(conv_out)
        #   conv_out = self.convattn(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred


# class EmbConvLstPoly6(nn.Module):
#     def __init__(self, note_data, dropout=0.3, bidirectional=False):
#         super(EmbConvLstPoly6, self).__init__()
#         self.dropout = dropout
#
#         self.bidirectional = bidirectional
#         self.note_data = note_data
#         self.embedding_size1 = 100
#         self.embedding_size2 = 100
#         self.embedding_size3 = 100
#         self.embedding_size4 = 100
#
#         self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
#         self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
#         self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
#         self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
#         self.comp_size = 512
#
#         self.note_encode = EmbEncode(600, 256)
#         self.offset_encode = EmbEncode(600, 256)
#         self.duration_encode = EmbEncode(600, 256)
#         self.velocity_encode = EmbEncode(600, 256)
#
#         self.linIn = nn.Linear(1024, 512)
#
#
#
#         self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=1, padding=0)
#         self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=3, padding=0)
#         self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=4, padding=0)
#         self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=8, padding=0)
#
#
#         self.lstm = nn.LSTM(self.comp_size * 4 * 1, self.comp_size * 2 * 1, 2, batch_first=True)
#        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)
#
#         self.linOut1 = nn.Linear(1024, 512)
#         self.linOut2 = nn.Linear(512, 1024)
#
#
#
#         self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab * 6)
#         self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab * 6)
#         self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab * 6)
#         self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab * 6)
#
#         self.linIn.weight = self.linOut1.weight
#         self._initialize_weights()
#
#     def init_hidden(self,device, batch_size=160):
#         hidden = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
#         cell = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
#         return hidden, cell
#
#
#     def detach_hidden(self, hidden):
#         hidden, cell = hidden
#         hidden = hidden.detach()
#         cell = cell.detach()
#         return hidden, cell
#
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, x, hidden):
#
#         notes = x[:, :, 0, :]
#         offsets = x[:, :, 1, :]
#         durations = x[:, :, 2, :]
#         velocities = x[:, :, 3, :]
#
#
# #dimenions of input x torch.Size([32, 64, 4, 6])
# #embedding shape not reshaped torch.Size([32, 64, 6])
#
#         note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1).permute(0, 2, 1)
#         offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1).permute(0, 2, 1)
#         duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1).permute(0, 2, 1)
#         velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1).permute(0, 2, 1)
#
#
#         note_embedd = self.note_encode(note_embedd)
#         offset_embedd = self.offset_encode(offset_embedd)
#         duration_embedd = self.duration_encode(duration_embedd)
#         velocity_embedd = self.velocity_encode(velocity_embedd)
#
#
#         embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=1)
#         #embeddings.permute(0,2,1)
#
#         #embeddings.permute(0, 2, 1)
#        # embeddings = embeddings.view(embeddings.shape[0], -1, embeddings.shape[2])
#         embeddings = embeddings.permute(0, 2, 1)  # shape [160, 32, 1024]
#         embeddings = embeddings.contiguous().view(-1, embeddings.shape[-1])  # shape [5120, 1024]
#         embeddings = self.linIn(embeddings)  # shape [5120, 512]
#         embeddings = embeddings.view(-1, 32, 512).permute(0, 2, 1)
#         embeddings = F.dropout(embeddings, self.dropout)
#
#         # Apply the convolutions separately to the input embeddings
#         conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
#         conv_out1 = F.mish(conv_out1)
#
#         padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
#         conv_out2 = self.conv2(padded_embeddings2)
#         conv_out2 = F.mish(conv_out2)
#
#         padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
#         conv_out3 = self.conv3(padded_embeddings3)
#         conv_out3 = F.mish(conv_out3)
#
#         padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
#         conv_out4 = self.conv4(padded_embeddings4)
#         conv_out4 = F.mish(conv_out4)
#
#         # Concatenate along the channel dimension
#         conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
#         #conv_out = self.convcomp(conv_out)
#         conv_out = conv_out.permute(0, 2, 1)
#         conv_out = F.mish(conv_out)
#      #   conv_out = self.convattn(conv_out)
#         conv_out = F.dropout(conv_out, self.dropout)
#         lstm_out, hidden  = self.lstm(conv_out, hidden)
#         lstm_out = lstm_out[:, -1, :]
#         lstm_out = self.linOut1(lstm_out)
#         lstm_out = self.linOut2(F.mish(lstm_out))
#
#         note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
#         offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
#         duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
#         velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
#         return note_pred, offset_pred, duration_pred, velocity_pred, hidden

class EmbEncode(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(EmbEncode, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, 1),
            #    SelfAttentionImpr(int(dim_in) / 2),
            #    nn.GroupNorm(6, int(dim_in / 2)),  # LayerNorm equivalent for 3D input
            nn.ReLU(),
            # nn.Conv1d(int(dim_in / 2), dim_out, 1),
            nn.GroupNorm(4, dim_out),  # LayerNorm equivalent for 3D input
            nn.ReLU()
        )

    def forward(self, x):
        return self.convs(x)


class EmbDecode(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(EmbDecode, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(dim_in, dim_in * 2, 1),
            nn.GroupNorm(32, int(dim_in * 2)),  # LayerNorm equivalent for 3D input
            nn.Mish(),
            nn.Conv1d(dim_in * 2, dim_out, 1),
            nn.GroupNorm(6, dim_out),  # LayerNorm equivalent for 3D input
            nn.Mish()
        )

    def forward(self, x):
        return self.convs(x)


class EmbConvLstPoly6(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly6, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 100
        self.embedding_size2 = 100
        self.embedding_size3 = 100
        self.embedding_size4 = 100

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.comp_size = 1024

        # self.note_encode = EmbEncode(600, 256)
        # self.offset_encode = EmbEncode(600, 256)
        # self.duration_encode = EmbEncode(600, 256)
        # self.velocity_encode = EmbEncode(600, 256)

        self.encode = EmbEncode(600 * 4, 1024)

        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=8, padding=0)
        self.conv5 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=16, padding=0)

        self.convAttn = SelfAttentionNorm(self.comp_size)

        self.lstm = nn.LSTM(self.comp_size * 1 * 1, self.comp_size * 1 * 1, 2, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        #  self.decode = EmbDecode(self.comp_size, self.comp_size * 4)

        self.fc_note = nn.Linear(self.comp_size * 1, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 1, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 1, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 1, note_data.v_vocab * 6)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 1).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 1).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):

        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1).permute(0, 2, 1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1).permute(0, 2, 1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1).permute(0, 2,
                                                                                                                   1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1).permute(0,
                                                                                                                      2,
                                                                                                                      1)

        #
        # note_embedd = self.note_encode(note_embedd)
        # offset_embedd = self.offset_encode(offset_embedd)
        # duration_embedd = self.duration_encode(duration_embedd)
        # velocity_embedd = self.velocity_encode(velocity_embedd)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=1)
        embeddings = self.encode(embeddings)
        # embeddings = embeddings.permute(0, 2, 1)  # shape [160, 32, 1024]
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        # conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        # conv_out1 = F.mish(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.mish(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.mish(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.mish(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv4(padded_embeddings4)
        conv_out5 = F.mish(conv_out5)

        # Concatenate along the channel dimension
        # conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = conv_out2 + conv_out3 + conv_out4 + conv_out5

        conv_out = conv_out.permute(0, 2, 1)
        conv_out_a = self.convAttn(conv_out)
        conv_out = torch.mul(conv_out, conv_out_a)
        conv_out = conv_out.permute(0, 2, 1)
        emb_comb = F.dropout(F.mish(embeddings), self.dropout)
        conv_out = torch.add(emb_comb / 2, conv_out)

        # conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = F.mish(conv_out)
        #   conv_out = self.convattn(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)
        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly65(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly65, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 8
        self.embedding_size2 = 8
        self.embedding_size3 = 8
        self.embedding_size4 = 8

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.comp_size = 8 * 4 * 4

        self.bn = nn.BatchNorm1d(8 * 4)
        self.embAttn = SelfAttentionImpr(8 * 4)
        self.dense = nn.Sequential(
            nn.Linear(8 * 4, 8 * 4 * 4),
            nn.Dropout(),
            nn.ReLU()
        )

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=8, padding=0)

        self.convAttn = SelfAttentionImpr(self.comp_size)
        self.convGroup = nn.BatchNorm1d(self.comp_size)  # LayerNorm equivalent for 3D input
        self.convAttn2 = SelfAttentionImpr(self.comp_size)
        self.lstm = nn.LSTM(self.comp_size * 1 * 1, self.comp_size * 2 * 1, 1, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab * 6)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=128):
        hidden = torch.zeros(1, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(1, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):

        notes = x[:, :, 0]
        offsets = x[:, :, 1]
        durations = x[:, :, 2]
        velocities = x[:, :, 3]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, velocity_embedd, offset_embedd, duration_embedd), dim=1)
        # embeddings = embeddings.permute(0, 2, 1)  # shape [160, 32, 1024]
        embeddings = F.dropout(embeddings, self.dropout)

        embeddings = self.bn(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.embAttn(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dense(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.mish(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.mish(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.mish(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.mish(conv_out4)

        # Concatenate along the channel dimension
        # conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = conv_out1 + conv_out2 + conv_out3 + conv_out4

        # conv_out = self.convGroup(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out_a = self.convAttn(conv_out)
        conv_out = torch.mul(conv_out, conv_out_a)

        emb_comb = F.dropout(F.mish(embeddings), self.dropout)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = torch.add(emb_comb, conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.convAttn2(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.convGroup(conv_out)

        # conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = F.mish(conv_out)
        #   conv_out = self.convattn(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)
        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly655(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly655, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 50
        self.embedding_size2 = 50
        self.embedding_size3 = 50
        self.embedding_size4 = 50

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.comp_size = 512

        self.note_encode = EmbEncode(300, 128)
        self.offset_encode = EmbEncode(300, 128)
        self.duration_encode = EmbEncode(300, 128)
        self.velocity_encode = EmbEncode(300, 128)

        self.embAttn = SelfAttentionImpr(self.comp_size)

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0),
            nn.GroupNorm(32, self.comp_size * 2),
            nn.LeakyReLU(),
            nn.Conv1d(self.comp_size * 2, self.comp_size * 1, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0),
            nn.GroupNorm(32, self.comp_size * 2),
            nn.LeakyReLU(),
            nn.Conv1d(self.comp_size * 2, self.comp_size * 1, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0),
            nn.GroupNorm(32, self.comp_size * 2),
            nn.LeakyReLU(),
            nn.Conv1d(self.comp_size * 2, self.comp_size * 1, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.fc_conv = nn.Sequential(
            nn.Linear(self.comp_size * 4, self.comp_size * 2),
            nn.LeakyReLU(),
            SelfAttentionImpr(self.comp_size * 2),
            Permute(0, 2, 1),
            nn.GroupNorm(32, self.comp_size * 2)
        )

        self.lstm = nn.LSTM(self.comp_size * 2 * 1, self.comp_size * 2 * 1, 2, dropout=dropout, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab * 6)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=96):
        hidden = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):

        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1).permute(0, 2, 1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1).permute(0, 2, 1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1).permute(0, 2,
                                                                                                                   1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1).permute(0,
                                                                                                                      2,
                                                                                                                      1)

        note_embedd = self.note_encode(note_embedd)
        offset_embedd = self.offset_encode(offset_embedd)
        duration_embedd = self.duration_encode(duration_embedd)
        velocity_embedd = self.velocity_encode(velocity_embedd)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=1)
        # embeddings = embeddings.permute(0, 2, 1)  # shape [160, 32, 1024]
        embeddings = F.dropout(embeddings, self.dropout)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.embAttn(embeddings)
        embeddings = embeddings.permute(0, 2, 1)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)

        conv_out = torch.cat((conv_out1, conv_out3, conv_out2, conv_out4), dim=1)
        conv_out = F.dropout(conv_out, self.dropout)
        conv_out = F.leaky_relu(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.fc_conv(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        lstm_out, hidden = self.lstm(conv_out)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.dropout(lstm_out, self.dropout)

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly656(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly656, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.comp_size = 768

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=3, padding=0),
            nn.GroupNorm(32, self.comp_size * 1),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=4, padding=0),
            nn.GroupNorm(32, self.comp_size * 1),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=8, padding=0),
            nn.GroupNorm(32, self.comp_size * 1),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        # self.fc_conv = nn.Sequential (
        #     nn.Linear(self.comp_size * 4, self.comp_size * 2),
        #     nn.LeakyReLU(),
        #     SelfAttentionImpr(self.comp_size * 2),
        #     Permute(0,2,1),
        #     nn.GroupNorm(32, self.comp_size * 2)
        # )

        self.convattn = SelfAttentionImpr(self.comp_size * 4)
        self.lstm = nn.LSTM(self.comp_size * 4 * 1, self.comp_size * 1 * 1, 2, dropout=dropout, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        self.fc_note = nn.Linear(self.comp_size * 1, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 1, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 1, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 1, note_data.v_vocab * 6)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 1).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 1).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):

        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1).permute(0, 2, 1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1).permute(0, 2, 1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1).permute(0, 2,
                                                                                                                   1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1).permute(0,
                                                                                                                      2,
                                                                                                                      1)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=1)
        # embeddings = embeddings.permute(0, 2, 1)  # shape [160, 32, 1024]
        embeddings = F.dropout(embeddings, self.dropout)
        # embeddings = embeddings.permute(0, 2, 1)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)

        # conv_out11 = torch.add(conv_out1, conv_out4)
        # conv_out22 = torch.add(conv_out4, conv_out3)
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = F.dropout(conv_out, self.dropout)
        conv_out = F.leaky_relu(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.convattn(conv_out)
        # conv_out = conv_out.permute(0, 2, 1)
        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.dropout(lstm_out, self.dropout)

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly7(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly7, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.note_comp = nn.Linear(384, 256)
        self.offset_comp = nn.Linear(384, 256)
        self.duration_comp = nn.Linear(384, 256)
        self.velocity_comp = nn.Linear(384, 256)

        self.comp_size = 1024

        # self.note_encode = EmbEncode(300, 128)
        # self.offset_encode = EmbEncode(300, 128)
        # self.duration_encode = EmbEncode(300, 128)
        # self.velocity_encode = EmbEncode(300, 128)

        # self.embeddattn = SelfAttentionNorm(1536)
        # self.comp = nn.Sequential(
        #     nn.Linear(1536, 768),
        #     nn.LayerNorm(768),
        #     nn.ReLU(),
        #     nn.Linear(768, 384),
        #     nn.LayerNorm(384),
        #     nn.ReLU(),
        #   #  nn.Linear(384, 256),
        #    # nn.LayerNorm(256),
        #    # nn.ReLU(),
        #     nn.Dropout(dropout),
        # )

        # self.comp = nn.Sequential(
        #     nn.Conv1d(1536, 768, 1),
        #    # SelfAttentionNorm(768),
        #     nn.BatchNorm1d(768),
        #     nn.ReLU(),
        #     nn.Conv1d(768, 384, 1),
        #     nn.BatchNorm1d(384),
        #     nn.ReLU(),
        #
        # )

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=8, padding=0)

        self.convAttn = SelfAttentionImpr(self.comp_size)
        #   self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=16, padding=0)
        # self.relu = nn.ReLU()

        # self.convattn = SelfAttentionNorm(self.comp_size * 4 * 1)

        # self.convcomp = nn.Sequential(
        #     nn.Conv1d(self.comp_size * 8, self.comp_size * 4, 1),
        #     nn.Mish(),
        #     Permute(0,2,1),
        #     SelfAttentionNorm(self.comp_size * 4),
        #     Permute(0,2,1),
        #     nn.Conv1d(self.comp_size * 4, self.comp_size * 2, 1),
        #     nn.Mish(),
        # )

        self.lstm = nn.LSTM(self.comp_size * 1, self.comp_size * 1, 2, batch_first=True)

        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        # self.decode = nn.Sequential(
        #     nn.ConvTranspose1d(self.comp_size * 4, self.comp_size * 5, 1, padding=0),  # Transposed convolutions
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(self.comp_size * 5, self.comp_size * 6, 1, padding=0),  # Transposed convolutions
        #     nn.ReLU(),
        # )

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.notedecomp = nn.Linear(self.comp_size, 256)
        self.fc_note = nn.Linear(256, note_data.n_vocab * 6)
        self.offsetdecomp = nn.Linear(self.comp_size, 256)
        self.fc_offset = nn.Linear(256, note_data.o_vocab * 6)
        self.durationdecomp = nn.Linear(self.comp_size, 256)
        self.fc_duration = nn.Linear(256, note_data.d_vocab * 6)
        self.velocitydecomp = nn.Linear(self.comp_size, 256)
        self.fc_velocity = nn.Linear(256, note_data.v_vocab * 6)

        self.fc_note.weight = self.note_comp.weight
        self.fc_offset.weight = self.offset_comp.weight
        self.fc_duration.weight = self.duration_comp.weight
        self.fc_velocity.weight = self.velocitydecomp.weight

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1)

        note_embedd = self.note_comp(note_embedd).permute(0, 2, 1)
        offset_embedd = self.offset_comp(offset_embedd).permute(0, 2, 1)
        duration_embedd = self.duration_comp(duration_embedd).permute(0, 2, 1)
        velocity_embedd = self.velocity_comp(velocity_embedd).permute(0, 2, 1)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=1)
        # embeddings = embeddings.view(embeddings.shape[0], -1, embeddings.shape[2])
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.mish(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.mish(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.mish(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.mish(conv_out4)

        # Concatenate along the channel dimension
        # conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = conv_out1 + conv_out2 + conv_out3 + conv_out4
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.convAttn(conv_out)
        conv_out = torch.mul(embeddings.permute(0, 2, 1), conv_out)

        conv_out = F.mish(conv_out)
        #   conv_out = self.convattn(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)
        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.dropout(lstm_out, self.dropout)

        note_pred = self.notedecomp(lstm_out)
        offset_pred = self.offsetdecomp(lstm_out)
        duration_pred = self.durationdecomp(lstm_out)
        velocity_pred = self.velocitydecomp(lstm_out)
        print(note_pred.shape)
        note_pred = self.fc_note(note_pred).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.fc_offset(offset_pred).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.fc_duration(duration_pred).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.fc_velocity(velocity_pred).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly8(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly8, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 128
        self.embedding_size2 = 128
        self.embedding_size3 = 128
        self.embedding_size4 = 128

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)

        self.comp_size = 1024

        self.note_encode = EmbEncode(768, 256)
        self.offset_encode = EmbEncode(768, 256)
        self.duration_encode = EmbEncode(768, 256)
        self.velocity_encode = EmbEncode(768, 256)

        self.linIn = nn.Linear(1024, 1024)

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=8, padding=0)

        self.lstm = nn.LSTM(self.comp_size * 4 * 1, self.comp_size * 2 * 1, 2, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        self.attention = SelfAttentionImpr(1024)
        self.linOut = nn.Linear(1024, 768)

        self.fc_note = nn.Linear(768, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(768, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(768, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(768, note_data.v_vocab * 6)

        # self.linIn.weight = self.linOut.weight
        # self.note_embedd.weight = self.fc_note.weight
        # self.offset_embedd.weight = self.fc_offset.weight
        # self.duration_embedd.weight = self.fc_duration.weight
        # self.velocity_embedd.weight = self.fc_velocity.weight

        self._initialize_weights()

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):

        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1).permute(0, 2, 1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1).permute(0, 2, 1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1).permute(0, 2,
                                                                                                                   1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1).permute(0,
                                                                                                                      2,
                                                                                                                      1)

        note_embedd = self.note_encode(note_embedd)
        offset_embedd = self.offset_encode(offset_embedd)
        duration_embedd = self.duration_encode(duration_embedd)
        velocity_embedd = self.velocity_encode(velocity_embedd)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=1)
        # embeddings.permute(0,2,1)

        # embeddings.permute(0, 2, 1)
        # embeddings = embeddings.view(embeddings.shape[0], -1, embeddings.shape[2])
        embeddings = embeddings.permute(0, 2, 1)  # shape [160, 32, 1024]
        embeddings = embeddings.contiguous().view(-1, embeddings.shape[-1])  # shape [5120, 1024]
        embeddings = self.linIn(embeddings)  # shape [5120, 512]
        embeddings = self.attention(embeddings)  # shape [5120, 512]

        embeddings = embeddings.view(-1, 32, 512).permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.mish(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.mish(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.mish(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.mish(conv_out4)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        # conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = F.mish(conv_out)
        #   conv_out = self.convattn(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)
        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.linOut1(F.mish(lstm_out))

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_data.n_vocab)  # Reshape output
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.note_data.o_vocab)  # Reshape output
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.note_data.d_vocab)  # Reshape output
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.note_data.v_vocab)  # Reshape output
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class MidiLSTM2D(nn.Module):
    def __init__(self, note_data):
        super(MidiLSTM2D, self).__init__()
        self.note_data = note_data
        self.note_embedd = nn.Embedding(note_data.n_vocab, 64, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, 64, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, 64, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, 64, padding_idx=0)

        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(1, 4), padding=0),  # using 2D convolution
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 4), padding=0),  # using 2D convolution
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(44428, 2048, 2, batch_first=True)  # update LSTM input size

        # update outout layer's input size
        self.fc_note = nn.Linear(2048, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(2048, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(2048, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(2048, note_data.v_vocab * 6)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, 2048).to(device)
        cell = torch.zeros(2, batch_size, 2048).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        embed_funcs = [self.note_embedd, self.offset_embedd, self.duration_embedd, self.velocity_embedd]
        embedd_views = []

        for i in range(4):
            embedd_views.append(embed_funcs[i](x[:, :, i, :]))

        embeddings = torch.stack(embedd_views, dim=1)
        embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1, embeddings.shape[-1])

        conv_out = self.conv(embeddings)

        conv_out = conv_out.view(conv_out.shape[0], conv_out.shape[1], -1)  # reshape for LSTM
        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class MidiLSTM2DA(nn.Module):
    def __init__(self, note_data, dropout=0.3):
        super(MidiLSTM2DA, self).__init__()
        self.max_vocab = note_data.n_vocab + note_data.o_vocab + note_data.d_vocab + note_data.v_vocab
        self.note_data = note_data
        self.note_embedd = nn.Embedding(note_data.n_vocab, 64)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, 64)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, 64)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, 64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(1, 4), padding=(0, 2)),  # using 2D convolution
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(1, 4), padding=(0, 4)),  # using 2D convolution
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 4), padding=(0, 4)),  # using 2D convolution
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=(1, 4), padding=(0, 2)),  # using 2D convolution
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(4, 4), padding=(0, 2)),  # using 2D convolution
            nn.ReLU(),

        )

        #  self.convcomp =  nn.Conv2d(24, 8, kernel_size=(1, 1), padding=0)  # using 2D convolution

        self.lstm = nn.LSTM(23424, 1024, 2, batch_first=True)

        # update outout layer's input size
        self.note_out = nn.Linear(1024, note_data.n_vocab * 6)
        self.offset_out = nn.Linear(1024, note_data.o_vocab * 6)
        self.duration_out = nn.Linear(1024, note_data.d_vocab * 6)
        self.velocity_out = nn.Linear(1024, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, 1024).to(device)
        cell = torch.zeros(2, batch_size, 1024).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        embed_funcs = [self.note_embedd, self.offset_embedd, self.duration_embedd, self.velocity_embedd]
        embedd_views = []

        for i in range(4):
            embedd_views.append(embed_funcs[i](x[:, :, i, :]))

        embeddings = torch.stack(embedd_views, dim=1)
        print("e1", embeddings.shape)
        embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1, embeddings.shape[-1])

        print("e2", embeddings.shape)

        print(embeddings.shape)

        conv_out1 = self.conv1(embeddings)
        conv_out2 = self.conv2(embeddings)
        conv_out3 = self.conv3(embeddings)

        conv_out = torch.cat((conv_out1, conv_out2, conv_out3), dim=1)
        conv_out = self.convcomp(conv_out)
        conv_out = conv_out.view(conv_out.shape[0], conv_out.shape[1], -1)  # reshape for LSTM
        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.relu(lstm_out)

        note_pred = self.note_out(lstm_out).view(-1, 6, self.note_data.n_vocab)
        offset_pred = self.offset_out(lstm_out).view(-1, 6, self.note_data.o_vocab)
        duration_pred = self.duration_out(lstm_out).view(-1, 6, self.note_data.d_vocab)
        velocity_pred = self.velocity_out(lstm_out).view(-1, 6, self.note_data.v_vocab)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class IMod(nn.Module):
    def __init__(self, chan_in, chan_out, dropout_rate=None):
        super(IMod, self).__init__()
        self.dropout = dropout_rate

        self.conv1 = nn.Conv1d(chan_in, chan_out, kernel_size=1, padding=0)

        self.conv2 = nn.Sequential(
            nn.Conv1d(chan_in, chan_out, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(chan_out, chan_out, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(chan_in, chan_out, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(chan_out, chan_out, kernel_size=5, padding=2),
            nn.LeakyReLU()
        )

        # dimensionality reduction
        # self.conv4 = nn.Conv1d(num_channels * 4, num_channels, kernel_size=1)

    def forward(self, x):
        conv_out1 = self.conv1(x)
        conv_out3 = self.conv32(x)
        conv_out4 = self.conv42(x)

        cat = torch.cat((conv_out1, conv_out3, conv_out4), dim=1)
        if (self.dropout is not None):
            cat = F.dropout(cat, self.dropout)
        return F.leaky_relu_(cat)


class NewSingleModel(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(NewSingleModel, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32
        self.bidirectional = bidirectional
        # self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4)
        self.e_size = 256
        self.comp_size = 128

        self.in_linear = nn.Sequential(
            SelfAttention(self.e_size),
            nn.LeakyReLU(),
            nn.Linear(self.e_size, self.comp_size),
        )

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=8, padding=0)

        self.incpt = nn.Sequential(
            IncepModule(self.comp_size, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepBottleNeckModule(self.comp_size * 4, self.comp_size * 2, self.comp_size * 8)
        )

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
        #                     bidirectional=bidirectional)
        # self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size* 8, 2, True)
        self.attn = SelfAttentionImpr(self.comp_size * 4)
        self.lstm = nn.LSTM(self.comp_size * 4, self.comp_size * 4, 2, batch_first=True)
        #  self.lstm = LSTM_ARCH(self.e_size, self.comp_size * 8, self.comp_size * 8 , lstm=self._lstm, dropout_rate=dropout)

        self.fc_out = nn.Sequential(
            # nn.Linear(self.comp_size * 8, self.comp_size * 4),
            # nn.LeakyReLU(),
            nn.Linear(self.comp_size * 4, self.comp_size * 2),
            nn.LeakyReLU(),
            nn.Linear(self.comp_size * 2, self.comp_size * 1),
            nn.LeakyReLU(),
        )
        self.dropout = nn.Dropout(dropout)

        self.fc_note = nn.Linear(self.e_size, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.e_size, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.e_size, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.e_size, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        return hidden, cell

    # return self.lstm.init_hidden(device, batch_size)

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

        # return self.lstm.detach_hidden(hidden)

    def forward(self, x, hidden):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        # embeddings = self.in_linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        incept_out = self.incpt(embeddings)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.leaky_relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.leaky_relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.leaky_relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.leaky_relu(conv_out4)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = torch.add(conv_out, incept_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = F.leaky_relu(conv_out)
        conv_out = self.attn(conv_out)

        conv_out = self.dropout(conv_out)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1]
        lstm_out = self.dropout(lstm_out)
        lstm_out = F.leaky_relu(lstm_out)
        lstm_out = self.fc_out(lstm_out)

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)
        output_velocity = self.fc_velocity(lstm_out)

        return output_note, output_offset, output_duration, output_velocity, hidden


class NewSingleModel2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(NewSingleModel2, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.dropout = dropout

        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64
        self.bidirectional = bidirectional
        # self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.e_size = 256
        self.comp_size = 128

        self.n_attn = nn.Sequential(
            #   nn.LayerNorm(64),
            # SelfAttention(64),
            # nn.LayerNorm(64)
        )

        self.o_attn = nn.Sequential(
            # nn.LayerNorm(64),
            # SelfAttention(64),
            # nn.LayerNorm(64)
        )
        self.d_attn = nn.Sequential(
            # nn.LayerNorm(64),
            # SelfAttention(64),
            #  nn.LayerNorm(64)
        )
        self.v_attn = nn.Sequential(
            #   nn.LayerNorm(64),
            # SelfAttention(64),
            #  nn.LayerNorm(64)
        )

        self.in_linear = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.e_size, self.comp_size),
        )

        self.incpt = nn.Sequential(
            IncepModule(self.comp_size, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size),
            IncepModule(self.comp_size * 4, self.comp_size * 2),
            IncepModule(self.comp_size * 8, self.comp_size * 2),
            IncepModule(self.comp_size * 8, self.comp_size),
            # IncepBottleNeckModule(self.comp_size * 4, self.comp_size * 2, self.comp_size * 10)
        )

        self.incpt2 = nn.Sequential(
            IncepModule2(self.comp_size, self.comp_size),
            IncepModule2(self.comp_size * 4, self.comp_size),
            IncepModule2(self.comp_size * 4, self.comp_size * 2),
            IncepModule2(self.comp_size * 8, self.comp_size * 2),
            IncepModule2(self.comp_size * 8, self.comp_size),
            # IncepBottleNeckModule(self.comp_size * 4, self.comp_size * 2, self.comp_size * 10)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=16, padding=0),
            nn.LeakyReLU()
        )

        self.attn1 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(self.comp_size * 4),
            # Permute(0, 2, 1)
        )
        self.attn2 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(self.comp_size * 4),
            # Permute(0, 2, 1)
        )
        self.attn3 = nn.Sequential(
            nn.BatchNorm1d(self.comp_size * 8),
            Permute(0, 2, 1),
            SelfAttention(self.comp_size * 8),
            # Permute(0, 2, 1)
        )

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
        #                     bidirectional=bidirectional)
        # self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size* 8, 2, True)
        self.lstm = nn.LSTM(self.comp_size * 8, self.comp_size * 2, 2, dropout=dropout, batch_first=True)
        #  self.lstm = LSTM_ARCH(self.e_size, self.comp_size * 8, self.comp_size * 8 , lstm=self._lstm, dropout_rate=dropout)
        #
        # self.fc_out = nn.Sequential(
        #     nn.Linear(self.comp_size * 8, self.comp_size * 4),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.comp_size * 4, self.comp_size * 2),
        #     nn.LeakyReLU(),
        #     # nn.Linear(self.comp_size * 2, self.comp_size * 1),
        #     # nn.LeakyReLU(),
        # )

        self.fc_nt = nn.Linear(self.comp_size * 2, 64)
        self.fc_ot = nn.Linear(self.comp_size * 2, 64)
        self.fc_dt = nn.Linear(self.comp_size * 2, 64)
        self.fc_vt = nn.Linear(self.comp_size * 2, 64)

        self.fc_note = nn.Linear(64, note_data.n_vocab)
        self.fc_offset = nn.Linear(64, note_data.o_vocab)
        self.fc_duration = nn.Linear(64, note_data.d_vocab)
        self.fc_velocity = nn.Linear(64, note_data.v_vocab)

        self._initialize_weights()

        self.note_embedding.weight = self.fc_note.weight
        self.offset_embedding.weight = self.fc_offset.weight
        self.duration_embedding.weight = self.fc_duration.weight
        self.velocity_embedding.weight = self.fc_velocity.weight

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    # return self.lstm.init_hidden(device, batch_size)

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

        # return self.lstm.detach_hidden(hidden)

    def forward(self, x, hidden):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        note_embedding = self.n_attn(note_embedding)
        offset_embedding = self.n_attn(offset_embedding)
        duration_embedding = self.n_attn(duration_embedding)
        velocity_embedding = self.n_attn(velocity_embedding)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = self.in_linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        incept_out = self.incpt(embeddings)
        incept_out2 = self.incpt2(embeddings)

        # Apply the convolutions separately to the input embeddings
        # conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        # conv_out1 = F.leaky_relu(conv_out1)
        #
        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.leaky_relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.leaky_relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.leaky_relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.leaky_relu(conv_out5)

        # Concatenate along the channel dimension
        incept_out = self.attn1(F.dropout(incept_out, self.dropout))
        incept_out2 = self.attn2(F.dropout(incept_out2, self.dropout))
        conv_out = torch.cat((incept_out, incept_out2), dim=2)
        conv_mul = torch.cat((conv_out2, conv_out3, conv_out4, conv_out5), dim=1)
        conv_mul = self.attn3(F.dropout(conv_mul, self.dropout))
        conv_out = torch.mul(conv_out, conv_mul)
        conv_out = F.leaky_relu(conv_out)
        #  conv_out = conv_out.permute(0, 2, 1)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1]
        lstm_out = F.dropout(lstm_out, self.dropout)
        lstm_out = F.leaky_relu(lstm_out)
        # lstm_out = self.fc_out(lstm_out)

        output_note = self.fc_nt(lstm_out)
        output_offset = self.fc_ot(lstm_out)
        output_duration = self.fc_dt(lstm_out)
        output_velocity = self.fc_vt(lstm_out)

        output_note = self.fc_note(output_note)
        output_offset = self.fc_offset(output_offset)
        output_duration = self.fc_duration(output_duration)
        output_velocity = self.fc_velocity(output_velocity)

        return output_note, output_offset, output_duration, output_velocity, hidden


class NewSingleModel3(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(NewSingleModel3, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.dropout = dropout

        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64
        self.bidirectional = bidirectional
        # self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.e_size = 256
        self.comp_size = 128

        self.n_attn = nn.Sequential(
            #   nn.LayerNorm(64),
            # SelfAttention(64),
            # nn.LayerNorm(64)
        )

        self.o_attn = nn.Sequential(
            # nn.LayerNorm(64),
            # SelfAttention(64),
            # nn.LayerNorm(64)
        )
        self.d_attn = nn.Sequential(
            # nn.LayerNorm(64),
            # SelfAttention(64),
            #  nn.LayerNorm(64)
        )
        self.v_attn = nn.Sequential(
            #   nn.LayerNorm(64),
            # SelfAttention(64),
            #  nn.LayerNorm(64)
        )

        self.in_linear = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.e_size, self.comp_size),
        )

        self.incpt = nn.Sequential(
            IncepModuleMidi(self.comp_size, self.comp_size),
            IncepModuleMidi(self.comp_size * 4, self.comp_size),
            IncepModuleMidi(self.comp_size * 4, self.comp_size * 2),
            IncepModuleMidi(self.comp_size * 8, self.comp_size * 2),
            IncepModuleMidi(self.comp_size * 8, self.comp_size * 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=16, padding=0),
            nn.LeakyReLU()
        )

        self.attn1 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(self.comp_size * 8),
            # Permute(0, 2, 1)
        )
        self.attn2 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(self.comp_size * 8),
            # Permute(0, 2, 1)
        )

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
        #                     bidirectional=bidirectional)
        # self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size* 8, 2, True)
        self.lstm = nn.LSTM(self.comp_size * 8, self.comp_size * 2, 2, dropout=dropout, batch_first=True)
        #  self.lstm = LSTM_ARCH(self.e_size, self.comp_size * 8, self.comp_size * 8 , lstm=self._lstm, dropout_rate=dropout)
        #
        # self.fc_out = nn.Sequential(
        #     nn.Linear(self.comp_size * 8, self.comp_size * 4),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.comp_size * 4, self.comp_size * 2),
        #     nn.LeakyReLU(),
        #     # nn.Linear(self.comp_size * 2, self.comp_size * 1),
        #     # nn.LeakyReLU(),
        # )

        self.fc_nt = nn.Linear(self.comp_size * 2, 64)
        self.fc_ot = nn.Linear(self.comp_size * 2, 64)
        self.fc_dt = nn.Linear(self.comp_size * 2, 64)
        self.fc_vt = nn.Linear(self.comp_size * 2, 64)

        self.fc_note = nn.Linear(64, note_data.n_vocab)
        self.fc_offset = nn.Linear(64, note_data.o_vocab)
        self.fc_duration = nn.Linear(64, note_data.d_vocab)
        self.fc_velocity = nn.Linear(64, note_data.v_vocab)

        self._initialize_weights()

        self.note_embedding.weight = self.fc_note.weight
        self.offset_embedding.weight = self.fc_offset.weight
        self.duration_embedding.weight = self.fc_duration.weight
        self.velocity_embedding.weight = self.fc_velocity.weight

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    # return self.lstm.init_hidden(device, batch_size)

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

        # return self.lstm.detach_hidden(hidden)

    def forward(self, x, hidden):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        note_embedding = self.n_attn(note_embedding)
        offset_embedding = self.n_attn(offset_embedding)
        duration_embedding = self.n_attn(duration_embedding)
        velocity_embedding = self.n_attn(velocity_embedding)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = self.in_linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        incept_out = self.incpt(embeddings)
        # incept_out2 = self.incpt2(embeddings)

        # Apply the convolutions separately to the input embeddings
        # conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        # conv_out1 = F.leaky_relu(conv_out1)
        #
        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.leaky_relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.leaky_relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.leaky_relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.leaky_relu(conv_out5)

        # Concatenate along the channel dimension
        incept_out = self.attn1(F.dropout(incept_out, self.dropout))
        conv_out = torch.cat((conv_out2, conv_out3, conv_out4, conv_out5), dim=1)
        conv_out = self.attn2(F.dropout(conv_out, self.dropout))
        conv_out = torch.add(conv_out, incept_out)
        conv_out = F.leaky_relu(conv_out)
        #  conv_out = conv_out.permute(0, 2, 1)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1]
        lstm_out = F.dropout(lstm_out, self.dropout)
        lstm_out = F.leaky_relu(lstm_out)
        # lstm_out = self.fc_out(lstm_out)

        output_note = self.fc_nt(lstm_out)
        output_offset = self.fc_ot(lstm_out)
        output_duration = self.fc_dt(lstm_out)
        output_velocity = self.fc_vt(lstm_out)

        output_note = self.fc_note(output_note)
        output_offset = self.fc_offset(output_offset)
        output_duration = self.fc_duration(output_duration)
        output_velocity = self.fc_velocity(output_velocity)

        return output_note, output_offset, output_duration, output_velocity, hidden


class NewSingleModel4(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(NewSingleModel4, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.dropout = dropout

        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64
        self.bidirectional = bidirectional
        # self.note_data = note_data

        self.note_embedding = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedding = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedding = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedding = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.e_size = 256
        self.comp_size = 128

        self.n_attn = nn.Sequential(
            #   nn.LayerNorm(64),
            # SelfAttention(64),
            # nn.LayerNorm(64)
        )

        self.o_attn = nn.Sequential(
            # nn.LayerNorm(64),
            # SelfAttention(64),
            # nn.LayerNorm(64)
        )
        self.d_attn = nn.Sequential(
            # nn.LayerNorm(64),
            # SelfAttention(64),
            #  nn.LayerNorm(64)
        )
        self.v_attn = nn.Sequential(
            #   nn.LayerNorm(64),
            # SelfAttention(64),
            #  nn.LayerNorm(64)
        )

        self.in_linear = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.e_size, self.comp_size),
        )

        self.incpt1 = nn.Sequential(
            IncepModuleMidi(self.comp_size, self.comp_size),
            IncepModuleMidi(self.comp_size * 4, self.comp_size),
            IncepModuleMidi(self.comp_size * 4, self.comp_size * 2),
            IncepModuleMidi(self.comp_size * 8, self.comp_size * 2)
        )

        self.batchnorm = nn.BatchNorm1d(self.comp_size * 8)

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0),
            nn.LeakyReLU()
        )

        self.attn1 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(self.comp_size * 8),
            Permute(0, 2, 1)
        )
        self.attn2 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(self.comp_size * 8),
            Permute(0, 2, 1)
        )

        # Adjust the input size of LSTM layer to take in account concatenated output from two conv layers
        # self.lstm = nn.LSTM(self.e_size * 2 * 5, self.e_size * 2 * 5, num_layers=2, batch_first=True, dropout=dropout,
        #                     bidirectional=bidirectional)
        # self._lstm = MultiLayerIdnRNN(self.comp_size * 8, self.comp_size* 8, 2, True)
        self.lstm = nn.LSTM(self.comp_size * 8, self.comp_size * 4, 2, dropout=dropout, batch_first=True)
        #  self.lstm = LSTM_ARCH(self.e_size, self.comp_size * 8, self.comp_size * 8 , lstm=self._lstm, dropout_rate=dropout)
        #
        self.fc_out = nn.Sequential(
            #     nn.Linear(self.comp_size * 8, self.comp_size * 4),
            nn.LeakyReLU(),
            nn.Linear(self.comp_size * 4, self.comp_size * 2),
            nn.LeakyReLU(),
            nn.Linear(self.comp_size * 2, self.comp_size * 1),
            nn.LeakyReLU(),
        )

        self.fc_nt = nn.Linear(self.comp_size, 64)
        self.fc_ot = nn.Linear(self.comp_size, 64)
        self.fc_dt = nn.Linear(self.comp_size, 64)
        self.fc_vt = nn.Linear(self.comp_size, 64)

        self.fc_note = nn.Linear(64, note_data.n_vocab)
        self.fc_offset = nn.Linear(64, note_data.o_vocab)
        self.fc_duration = nn.Linear(64, note_data.d_vocab)
        self.fc_velocity = nn.Linear(64, note_data.v_vocab)

        self._initialize_weights()

        self.note_embedding.weight = self.fc_note.weight
        self.offset_embedding.weight = self.fc_offset.weight
        self.duration_embedding.weight = self.fc_duration.weight
        self.velocity_embedding.weight = self.fc_velocity.weight

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        return hidden, cell

    # return self.lstm.init_hidden(device, batch_size)

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

        # return self.lstm.detach_hidden(hidden)

    def forward(self, x, hidden):
        batch_size, seq_length, _ = x.size()
        note = x[:, :, 0].long()
        offset = x[:, :, 1].long()
        duration = x[:, :, 2].long()
        velocity = x[:, :, 3].long()

        # print("Min value in note:", note.min())
        # print("Max value in note:", note.max())
        # print("Note vocabulary size:", self.note_data.n_vocab)
        note_embedding = self.note_embedding(note)
        offset_embedding = self.offset_embedding(offset)
        duration_embedding = self.duration_embedding(duration)
        velocity_embedding = self.velocity_embedding(velocity)

        note_embedding = self.n_attn(note_embedding)
        offset_embedding = self.n_attn(offset_embedding)
        duration_embedding = self.n_attn(duration_embedding)
        velocity_embedding = self.n_attn(velocity_embedding)

        embeddings = torch.cat((note_embedding, offset_embedding, duration_embedding, velocity_embedding), dim=2)
        embeddings = self.in_linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        incept_out = self.incpt1(embeddings)

        # incept_out2 = self.incpt2(embeddings)

        # Apply the convolutions separately to the input embeddings
        # conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        # conv_out1 = F.leaky_relu(conv_out1)
        #
        conv_out2 = self.conv2(embeddings)
        conv_out2 = F.leaky_relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings,
                                   (1, 1))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.leaky_relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings,
                                   (1, 2))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.leaky_relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings,
                                   (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.leaky_relu(conv_out5)

        # Concatenate along the channel dimension

        conv_out = torch.cat((conv_out2, conv_out3, conv_out4, conv_out5), dim=1)
        conv_out = self.batchnorm(conv_out)
        conv_out = self.attn2(F.dropout(conv_out, self.dropout))
        incept_out = self.attn1(F.dropout(incept_out, self.dropout))
        conv_out = torch.mul(conv_out, incept_out / 2)

        conv_out = conv_out.permute(0, 2, 1)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1]
        lstm_out = F.dropout(lstm_out, self.dropout)
        lstm_out = F.leaky_relu(lstm_out)
        lstm_out = self.fc_out(lstm_out)

        output_note = self.fc_nt(lstm_out)
        output_offset = self.fc_ot(lstm_out)
        output_duration = self.fc_dt(lstm_out)
        output_velocity = self.fc_vt(lstm_out)

        output_note = self.fc_note(output_note)
        output_offset = self.fc_offset(output_offset)
        output_duration = self.fc_duration(output_duration)
        output_velocity = self.fc_velocity(output_velocity)

        return output_note, output_offset, output_duration, output_velocity, hidden


class EmbConvLstPoly(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstPoly, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64
        self.bidirectional = bidirectional
        # self.note_data = note_data
        self.dropout = dropout

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4)
        self.e_size = 256
        self.comp_size = 256
        self.comp = nn.Sequential(
            nn.Linear(self.comp_size * 6, self.comp_size * 4),
            nn.ReLU(),
            nn.Linear(self.comp_size * 4, self.comp_size * 2),
            nn.ReLU(),
            nn.Linear(self.comp_size * 2, self.comp_size),
            nn.ReLU(),
            # Permute(0,2,1)
        )

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=7, padding=0)
        self.conv5 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=8, padding=0)
        self.conv6 = nn.Conv1d(self.comp_size, self.comp_size, kernel_size=16, padding=0)

        # self.comp2 = nn.Sequential(
        #     Permute(0,2,1),
        #     SelfAttention(self.comp_size * 6),
        #     nn.Linear(self.comp_size * 6, self.comp_size * 4),
        #     nn.ReLU(),
        #     nn.Linear(self.comp_size * 4 , self.comp_size * 2),
        #     SelfAttention(self.comp_size * 2),
        #     nn.ReLU(),
        # )

        self.comp2 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(self.comp_size * 6),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.comp_size * 6, self.comp_size * 4, 2, batch_first=True)

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.fc_note = nn.Linear(self.comp_size * 4, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 4, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 4, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 4, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        embeddings = self.comp(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 3))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings, (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out6 = self.conv6(padded_embeddings6)
        conv_out6 = F.relu(conv_out6)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = F.dropout(conv_out, self.dropout)
        conv_out = F.relu(conv_out)
        conv_out = self.comp2(conv_out)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        # print(lstm_out.shape)

        # lstm_out = lstm_out[:, -1, :]
        # lstm_out = lstm_out[:, -1, :]
        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly2(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstPoly2, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64
        self.bidirectional = bidirectional
        # self.note_data = note_data
        self.dropout = dropout

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4)

        self.e_size = 256
        self.comp_size = 256

        self.combined_embedd = nn.Embedding(4 * self.e_size, self.comp_size)

        # self.comp = nn.Sequential(
        #     nn.Linear(self.comp_size * 6, self.comp_size * 4),
        #     nn.ReLU(),
        #     nn.Linear(self.comp_size * 4 , self.comp_size * 2),
        #     nn.ReLU(),
        #     nn.Linear(self.comp_size * 2, self.comp_size),
        #     nn.ReLU(),
        #     #Permute(0,2,1)
        # )

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=7, padding=0)
        self.conv5 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0)
        self.conv6 = nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=16, padding=0)

        self.comp2 = nn.Sequential(
            Permute(0, 2, 1),
            nn.Linear(self.comp_size * 12, self.comp_size * 8),
            nn.ReLU(),
            nn.Linear(self.comp_size * 8, self.comp_size * 6),
            nn.ReLU(),
            SelfAttention(self.comp_size * 6)
        )

        self.lstm = nn.LSTM(self.comp_size * 6, self.comp_size * 4, 2, batch_first=True)

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.fc_note = nn.Linear(self.comp_size * 4, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 4, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 4, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 4, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 4).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        embeddings_flattened = embeddings.view(embeddings.shape[0], embeddings.shape[1], -1)
        embeddings = self.combined_embedd(embeddings_flattened)
        # embeddings = self.comp(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 3))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings, (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out6 = self.conv6(padded_embeddings6)
        conv_out6 = F.relu(conv_out6)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6), dim=1)
        conv_out = F.dropout(conv_out, self.dropout)
        conv_out = F.relu(conv_out)
        conv_out = self.comp2(conv_out)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        # print(lstm_out.shape)

        # lstm_out = lstm_out[:, -1, :]
        # lstm_out = lstm_out[:, -1, :]
        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly3(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstPoly3, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 16
        self.embedding_size2 = 16
        self.embedding_size3 = 16
        self.embedding_size4 = 16
        self.bidirectional = bidirectional
        # self.note_data = note_data
        self.dropout = dropout

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.e_size = 64
        self.comp_size = 64
        self.comp = nn.Sequential(
            nn.Linear(self.comp_size * 6, self.comp_size * 3),
            nn.ReLU(),
            # Permute(0,2,1)
        )

        self.conv1 = nn.Conv1d(self.comp_size * 3, self.comp_size * 3, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size * 3, self.comp_size * 3, kernel_size=3, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size * 3, self.comp_size * 3, kernel_size=4, padding=0)
        self.conv5 = nn.Conv1d(self.comp_size * 3, self.comp_size * 3, kernel_size=8, padding=0)

        # self.comp2 = nn.Sequential(
        #     Permute(0,2,1),
        #     SelfAttention(self.comp_size * 6),
        #     nn.Linear(self.comp_size * 6, self.comp_size * 4),
        #     nn.ReLU(),
        #     nn.Linear(self.comp_size * 4 , self.comp_size * 2),
        #     SelfAttention(self.comp_size * 2),
        #     nn.ReLU(),
        # )

        self.comp2 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(self.comp_size * 12),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.comp_size * 12, self.comp_size * 6, 2, batch_first=True)

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.fc_note = nn.Linear(self.comp_size * 6, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 6, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 6, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 6, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(2, batch_size, self.comp_size * 6).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 6).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        embeddings = self.comp(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings,
                                   (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        # padded_embeddings4 = F.pad(embeddings,
        #                            (3, 3))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        # conv_out4 = self.conv4(padded_embeddings4)
        # conv_out4 = F.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings,
                                   (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.relu(conv_out5)

        # padded_embeddings6 = F.pad(embeddings,
        #                            (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        # conv_out6 = self.conv6(padded_embeddings6)
        # conv_out6 = F.relu(conv_out6)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out5,), dim=1)
        conv_out = F.relu(conv_out)
        conv_out = self.comp2(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        # print(lstm_out.shape)

        # lstm_out = lstm_out[:, -1, :]
        # lstm_out = lstm_out[:, -1, :]
        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly5(nn.Module):
    def __init__(self, note_data, dropout=0.7, bidirectional=False):
        super(EmbConvLstPoly5, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 64
        self.embedding_size2 = 64
        self.embedding_size3 = 64
        self.embedding_size4 = 64
        self.bidirectional = bidirectional
        # self.note_data = note_data
        self.dropout = dropout

        # self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        # self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        # self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        # self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4)
        self.e_size = 64 * 4 * 6
        self.comp_size = 24 * 4 * 6

        self.conv1 = nn.Conv1d(self.e_size, self.comp_size, kernel_size=1, padding=0)

        self.conv2 = nn.Conv1d(self.comp_size, int(self.comp_size), kernel_size=3, padding=0)

        self.conv3 = nn.Conv1d(self.comp_size, int(self.comp_size), kernel_size=4, padding=0)

        self.conv5 = nn.Conv1d(self.comp_size, int(self.comp_size), kernel_size=8, padding=0)

        self.conv6 = nn.Conv1d(self.comp_size, int(self.comp_size), kernel_size=7, padding=0)

        #
        # self.comp2 = nn.Sequential(
        #     Permute(0,2,1),
        #     SelfAttention(self.comp_size * 2),
        #     nn.Linear(self.comp_size * 2, self.comp_size),
        #     nn.ReLU(),
        #     nn.Linear(self.comp_size * 4 , self.comp_size * 2),
        #     SelfAttention(self.comp_size * 2),
        #     nn.ReLU(),
        # )

        self.comp2 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(int(self.comp_size * 4)),
            Permute(0, 2, 1),
            nn.Conv1d(self.comp_size * 4, int(self.comp_size), kernel_size=4, padding=0),
            Permute(0, 2, 1),
            nn.ReLU(),
            # nn.Linear(self.comp_size * 12, self.comp_size * 6),
            # SelfAttention(self.comp_size * 6),
            #  nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.comp_size, self.comp_size, 1, batch_first=True)

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.fc_note = nn.Linear(self.comp_size, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=160):
        hidden = torch.zeros(1, batch_size, self.comp_size).to(device)
        cell = torch.zeros(1, batch_size, self.comp_size).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        #  embeddings = self.comp(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.relu(conv_out1)

        # padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(F.pad(conv_out1, (1, 1)))
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings,
                                   (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(F.pad(conv_out1, (1, 2)))
        conv_out3 = F.relu(conv_out3)

        # padded_embeddings4 = F.pad(embeddings,
        #                            (3, 3))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        # conv_out4 = self.conv4(padded_embeddings4)
        # conv_out4 = F.relu(conv_out4)

        # padded_embeddings5 = F.pad(embeddings,
        #                            (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(F.pad(conv_out1, (3, 4)))
        conv_out5 = F.relu(conv_out5)

        # padded_embeddings6 = F.pad(embeddings,
        #                            (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out6 = self.conv6(F.pad(conv_out1, (3, 3)))
        conv_out6 = F.relu(conv_out6)

        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out5, conv_out2, conv_out3, conv_out6), dim=1)
        conv_out = F.relu(conv_out)
        conv_out = self.comp2(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        # print(lstm_out.shape)

        # lstm_out = lstm_out[:, -1, :]
        # lstm_out = lstm_out[:, -1, :]
        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


# class EmbConvLstPoly5(nn.Module):
#     def __init__(self, note_data, dropout=0.5, bidirectional=False):
#         super(EmbConvLstPoly5, self).__init__()
#         pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))
#
#         self.bidirectional = bidirectional
#         self.note_data = note_data
#
#         self.embedding_size1 = 16
#         self.embedding_size2 = 16
#         self.embedding_size3 = 16
#         self.embedding_size4 = 16
#         self.bidirectional = bidirectional
#         # self.note_data = note_data
#         self.dropout = dropout
#
#         self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1)
#         self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2)
#         self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3)
#         self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4)
#         self.e_size = 128 * 4
#         self.comp_size = 64 * 6
#
#
#         self.conv1 = nn.Conv1d(self.comp_size, int(self.comp_size ), kernel_size=1, padding=0)
#
#         self.conv2 = nn.Conv1d(self.comp_size,int(self.comp_size / 4), kernel_size=3, padding=0)
#         self.conv3 = nn.Conv1d(self.comp_size, int(self.comp_size / 4), kernel_size=4, padding=0)
#         self.conv5 = nn.Conv1d(self.comp_size, int(self.comp_size / 4), kernel_size=8, padding=0)
#         self.conv6 = nn.Conv1d(self.comp_size, int(self.comp_size / 4), kernel_size=16, padding=0)
#
#         #
#         # self.comp2 = nn.Sequential(
#         #     Permute(0,2,1),
#         #     SelfAttention(self.comp_size * 6),
#         #     nn.Linear(self.comp_size * 6, self.comp_size * 4),
#         #     nn.ReLU(),
#         #     nn.Linear(self.comp_size * 4 , self.comp_size * 2),
#         #     SelfAttention(self.comp_size * 2),
#         #     nn.ReLU(),
#         # )
#
#
#
#         self.comp2 = nn.Sequential(
#             Permute(0, 2, 1),
#             SelfAttention(self.comp_size * 2),
#             nn.ReLU(),
#            # nn.Linear(self.comp_size * 12, self.comp_size * 6),
#            #SelfAttention(self.comp_size * 6),
#           #  nn.ReLU(),
#         )
#
#         self.lstm = nn.LSTM(self.comp_size * 2, self.comp_size , 2, batch_first=True)
#
#         # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)
#
#         self.fc_note = nn.Linear(self.comp_size , note_data.n_vocab * 6)
#         self.fc_offset = nn.Linear(self.comp_size , note_data.o_vocab * 6)
#         self.fc_duration = nn.Linear(self.comp_size , note_data.d_vocab * 6)
#         self.fc_velocity = nn.Linear(self.comp_size , note_data.v_vocab * 6)
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def init_hidden(self, device, batch_size=160):
#         hidden = torch.zeros(2, batch_size, self.comp_size ).to(device)
#         cell = torch.zeros(2, batch_size, self.comp_size ).to(device)
#         return hidden, cell
#
#     def detach_hidden(self, hidden):
#         hidden, cell = hidden
#         hidden = hidden.detach()
#         cell = cell.detach()
#         return hidden, cell
#
#     def forward(self, x, hidden):
#         notes = x[:, :, 0, :]
#         offsets = x[:, :, 1, :]
#         durations = x[:, :, 2, :]
#         velocities = x[:, :, 3, :]
#
#         # dimenions of input x torch.Size([32, 64, 4, 6])
#         # embedding shape not reshaped torch.Size([32, 64, 6])
#
#         note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1)
#         offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1)
#         duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1)
#         velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1)
#
#         embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
#       #  embeddings = self.comp(embeddings)
#         embeddings = embeddings.permute(0, 2, 1)
#         embeddings = F.dropout(embeddings, self.dropout)
#
#         # Apply the convolutions separately to the input embeddings
#         conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
#         conv_out1 = F.relu(conv_out1)
#
#         padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
#         conv_out2 = self.conv2(padded_embeddings2)
#         conv_out2 = F.relu(conv_out2)
#
#         padded_embeddings3 = F.pad(embeddings,
#                                    (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
#         conv_out3 = self.conv3(padded_embeddings3)
#         conv_out3 = F.relu(conv_out3)
#
#         # padded_embeddings4 = F.pad(embeddings,
#         #                            (3, 3))  # Add 1 padding to left and 2 padding to right for kernel_size=4
#         # conv_out4 = self.conv4(padded_embeddings4)
#         # conv_out4 = F.relu(conv_out4)
#
#         padded_embeddings5 = F.pad(embeddings,
#                                    (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
#         conv_out5 = self.conv5(padded_embeddings5)
#         conv_out5 = F.relu(conv_out5)
#
#         padded_embeddings6 = F.pad(embeddings,
#                                    (7, 8))  # Add 3 padding to left and 4 padding to right for kernel_size=8
#         conv_out6 = self.conv6(padded_embeddings6)
#         conv_out6 = F.relu(conv_out6)
#
#         # Concatenate along the channel dimension
#         conv_out = torch.cat(( conv_out2, conv_out3, conv_out5, conv_out6), dim=1)
#         conv_out = F.relu(conv_out)
#         conv_out = torch.cat((conv_out1, conv_out), dim=1)
#         conv_out = F.relu(conv_out)
#         conv_out = self.comp2(conv_out)
#         conv_out = F.dropout(conv_out, self.dropout)
#
#
#
#         lstm_out, hidden = self.lstm(conv_out, hidden)
#        # lstm_out = F.dropout(conv_out, self.dropout)
#         lstm_out = lstm_out[:, -1, :]
#         # print(lstm_out.shape)
#
#         # lstm_out = lstm_out[:, -1, :]
#         # lstm_out = lstm_out[:, -1, :]
#         note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
#         offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
#         duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
#         velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
#         return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly10(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstPoly10, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32
        self.bidirectional = bidirectional
        # self.note_data = note_data
        self.dropout = dropout

        # self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        # self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        # self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        # self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4)
        self.e_size = 32 * 4 * 6
        self.comp_size = 32 * 4 * 6

        self.comp1 = nn.Sequential(
            nn.Linear(self.comp_size, int(self.comp_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.comp_size / 2), int(self.comp_size / 4)),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(int(self.comp_size / 4), int(self.comp_size / 2), kernel_size=3, padding=0),
            nn.Conv1d(int(self.comp_size / 2), int(self.comp_size / 2), kernel_size=1, padding=0)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(int(self.comp_size / 4), int(self.comp_size / 2), kernel_size=4, padding=0),
            nn.Conv1d(int(self.comp_size / 2), int(self.comp_size / 2), kernel_size=1, padding=0)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(int(self.comp_size / 4), int(self.comp_size / 2), kernel_size=8, padding=0),
            nn.Conv1d(int(self.comp_size / 2), int(self.comp_size / 2), kernel_size=1, padding=0)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(int(self.comp_size / 4), int(self.comp_size / 2), kernel_size=7, padding=0),
            nn.Conv1d(int(self.comp_size / 2), int(self.comp_size / 2), kernel_size=1, padding=0)
        )

        #
        # self.comp2 = nn.Sequential(
        #     Permute(0,2,1),
        #     SelfAttention(self.comp_size * 2),
        #     nn.Linear(self.comp_size * 2, self.comp_size),
        #     nn.ReLU(),
        #     nn.Linear(self.comp_size * 4 , self.comp_size * 2),
        #     SelfAttention(self.comp_size * 2),
        #     nn.ReLU(),
        # )

        self.comp2 = nn.Sequential(
            Permute(0, 2, 1),
            SelfAttention(int(self.comp_size * 2)),
            nn.Linear(self.comp_size * 2, self.comp_size * 1),
            nn.ReLU(),
            # SelfAttention(int(self.comp_size * 1)),
            # nn.Linear(self.comp_size * 1, int(self.comp_size /2)),
            # nn.ReLU(),

            # nn.Linear(self.comp_size * 12, self.comp_size * 6),
            # SelfAttention(self.comp_size * 6),
            #  nn.ReLU(),
        )

        self.lstm = nn.LSTM(int(self.comp_size), self.comp_size * 2, 1, batch_first=True)

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=256):
        hidden = torch.zeros(1, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(1, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        # embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.comp1(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        # conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        # conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings,
                                   (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        # padded_embeddings4 = F.pad(embeddings,
        #                            (3, 3))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        # conv_out4 = self.conv4(padded_embeddings4)
        # conv_out4 = F.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings,
                                   (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings,
                                   (3, 3))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out6 = self.conv6(padded_embeddings6)
        conv_out6 = F.relu(conv_out6)

        # Concatenate along the channel dimension
        # conv_out 5 was first.
        conv_out = torch.cat((conv_out2, conv_out3, conv_out5, conv_out6), dim=1)
        conv_out = F.relu(conv_out)
        conv_out = self.comp2(conv_out)
        #   conv_out = F.dropout(conv_out, self.dropout)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.dropout(lstm_out, self.dropout)
        # print(lstm_out.shape)

        # lstm_out = lstm_out[:, -1, :]
        # lstm_out = lstm_out[:, -1, :]
        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


# class EmbConvLstPoly5(nn.Module):


class EmbConvLstPoly11(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(EmbConvLstPoly11, self).__init__()
        pow2 = lambda n: 2 ** math.ceil(math.log(n, 2))

        self.bidirectional = bidirectional
        self.note_data = note_data

        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32
        self.bidirectional = bidirectional
        # self.note_data = note_data
        self.dropout = dropout

        # self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        # self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        # self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        # self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4)
        self.e_size = 32 * 4 * 6
        self.comp_size = 32 * 4 * 6

        self.embAttn = SelfAttentionImpr(self.comp_size)

        self.comp1 = nn.Sequential(
            nn.Linear(self.comp_size, int(self.comp_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.comp_size / 2), int(self.comp_size / 4)),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(int(self.comp_size / 4), int(self.comp_size / 2), kernel_size=3, padding=0),
            nn.Conv1d(int(self.comp_size / 2), int(self.comp_size / 2), kernel_size=1, padding=0)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(int(self.comp_size / 4), int(self.comp_size / 2), kernel_size=4, padding=0),
            nn.Conv1d(int(self.comp_size / 2), int(self.comp_size / 2), kernel_size=1, padding=0)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(int(self.comp_size / 4), int(self.comp_size / 2), kernel_size=8, padding=0),
            nn.Conv1d(int(self.comp_size / 2), int(self.comp_size / 2), kernel_size=1, padding=0)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(int(self.comp_size / 4), int(self.comp_size / 2), kernel_size=7, padding=0),
            nn.Conv1d(int(self.comp_size / 2), int(self.comp_size / 2), kernel_size=1, padding=0)
        )

        self.comp2 = nn.Sequential(
            nn.GroupNorm(32, self.comp_size * 2),
            Permute(0, 2, 1),
            nn.Linear(self.comp_size * 2, self.comp_size * 1),
            nn.ReLU(),

            # nn.Linear(self.comp_size * 12, self.comp_size * 6),
            # SelfAttention(self.comp_size * 6),
            #  nn.ReLU(),
        )

        self.conv_attn = SelfAttention(self.comp_size)

        self.lstm = nn.LSTM(int(self.comp_size), self.comp_size * 4, 1, batch_first=True)

        # self.fc_out = nn.Linear(self.comp_size * 8, self.e_size)

        self.fc_note = nn.Linear(self.comp_size * 4, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 4, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 4, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 4, note_data.v_vocab * 6)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=96):
        hidden = torch.zeros(1, batch_size, self.comp_size * 4).to(device)
        cell = torch.zeros(1, batch_size, self.comp_size * 4).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=-1)
        # embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.embAttn(embeddings)
        embeddings = self.comp1(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.dropout(embeddings, self.dropout)

        # Apply the convolutions separately to the input embeddings
        # conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        # conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings,
                                   (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        # padded_embeddings4 = F.pad(embeddings,
        #                            (3, 3))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        # conv_out4 = self.conv4(padded_embeddings4)
        # conv_out4 = F.relu(conv_out4)

        padded_embeddings5 = F.pad(embeddings,
                                   (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out5 = self.conv5(padded_embeddings5)
        conv_out5 = F.relu(conv_out5)

        padded_embeddings6 = F.pad(embeddings,
                                   (3, 3))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out6 = self.conv6(padded_embeddings6)
        conv_out6 = F.relu(conv_out6)

        # Concatenate along the channel dimension
        # conv_out 5 was first.
        conv_out = torch.cat((conv_out2, conv_out3, conv_out5, conv_out6), dim=1)
        conv_out = F.relu(conv_out)
        conv_out = self.comp2(conv_out)
        conv_out = self.conv_attn(conv_out)
        #   conv_out = F.dropout(conv_out, self.dropout)

        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.dropout(lstm_out, self.dropout)
        # print(lstm_out.shape)

        # lstm_out = lstm_out[:, -1, :]
        # lstm_out = lstm_out[:, -1, :]
        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


# class EmbConvLstPoly5(nn.Module):


class EmbConvLstPolyNew(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPolyNew, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 24
        self.embedding_size2 = 24
        self.embedding_size3 = 24
        self.embedding_size4 = 24

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.comp_size = 48 * 4

        self.note_encode = EmbEncode(24 * 6, 48)
        self.offset_encode = EmbEncode(24 * 6, 48)
        self.duration_encode = EmbEncode(24 * 6, 48)
        self.velocity_encode = EmbEncode(24 * 6, 48)

        self.fc_in = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=1, padding=0)

        self.comp_size = int(self.comp_size / 2)
        self.embAttn = SelfAttentionImpr(self.comp_size)

        self.conv1 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=4, padding=0)
        self.conv3 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=7, padding=0)
        self.conv4 = nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=8, padding=0)

        self.convAttn = SelfAttentionImpr(self.comp_size)
        self.convGroup = nn.GroupNorm(32, self.comp_size)  # LayerNorm equivalent for 3D input
        self.convAttn2 = SelfAttentionImpr(self.comp_size)
        self.lstm = nn.LSTM(self.comp_size * 1 * 1, self.comp_size * 2 * 1, 2, batch_first=True, dropout=0.2)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab * 6)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=512):
        hidden = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):

        notes = x[:, :, 0, :]
        offsets = x[:, :, 1, :]
        durations = x[:, :, 2, :]
        velocities = x[:, :, 3, :]

        # dimenions of input x torch.Size([32, 64, 4, 6])
        # embedding shape not reshaped torch.Size([32, 64, 6])

        note_embedd = self.note_embedd(notes).view(notes.shape[0], notes.shape[1], -1).permute(0, 2, 1)
        offset_embedd = self.offset_embedd(offsets).view(offsets.shape[0], offsets.shape[1], -1).permute(0, 2, 1)
        duration_embedd = self.duration_embedd(durations).view(durations.shape[0], durations.shape[1], -1).permute(0, 2,
                                                                                                                   1)
        velocity_embedd = self.velocity_embedd(velocities).view(velocities.shape[0], velocities.shape[1], -1).permute(0,
                                                                                                                      2,
                                                                                                                      1)

        note_embedd = self.note_encode(note_embedd)
        offset_embedd = self.offset_encode(offset_embedd)
        duration_embedd = self.duration_encode(duration_embedd)
        velocity_embedd = self.velocity_encode(velocity_embedd)

        embeddings = torch.cat((note_embedd, velocity_embedd, offset_embedd, duration_embedd), dim=1)
        # embeddings = embeddings.permute(0, 2, 1)  # shape [160, 32, 1024]
        embeddings = F.dropout(embeddings, self.dropout)

        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.fc_in(embeddings)
        embeddings = self.embAttn(embeddings)
        embeddings = embeddings.permute(0, 2, 1)

        # Apply the convolutions separately to the input embeddings
        padded_embeddings2 = F.pad(embeddings, (1, 1))
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1
        conv_out1 = F.relu(conv_out1)

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)
        conv_out2 = F.relu(conv_out2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)
        conv_out3 = F.relu(conv_out3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)
        conv_out4 = F.relu(conv_out4)

        # Concatenate along the channel dimension
        # conv_out = torch.cat((conv_out1, conv_out2, conv_out3, conv_out4), dim=1)
        conv_out = conv_out1 + conv_out2 + conv_out3 + conv_out4

        # conv_out = self.convGroup(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out_a = self.convAttn(conv_out)
        conv_out = torch.mul(conv_out, conv_out_a)

        emb_comb = F.dropout(F.relu(embeddings), self.dropout)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = torch.add(emb_comb, conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.convAttn2(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.convGroup(conv_out)

        # conv_out = self.convcomp(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = F.relu(conv_out)
        #   conv_out = self.convattn(conv_out)
        conv_out = F.dropout(conv_out, self.dropout)
        lstm_out, hidden = self.lstm(conv_out, hidden)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.fc_note(lstm_out).view(-1, 6, self.note_embedd.weight.size(0))
        velocity_pred = self.fc_velocity(lstm_out).view(-1, 6, self.velocity_embedd.weight.size(0))
        offset_pred = self.fc_offset(lstm_out).view(-1, 6, self.offset_embedd.weight.size(0))
        duration_pred = self.fc_duration(lstm_out).view(-1, 6, self.duration_embedd.weight.size(0))

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


# class SimpleLstm(nn.Module):
#     def __init__(self, note_data, dropout=0.3, bidirectional=False):
#         super(SimpleLstm, self).__init__()
#         self.dropout = dropout
#
#         self.bidirectional = bidirectional
#         self.note_data = note_data
#         self.embedding_size1 = 64
#         self.embedding_size2 = 128
#         self.embedding_size3 = 128
#         self.embedding_size4 = 64
#
#         self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
#         self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
#         self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
#         self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
#         self.in_size = 96 * 6 * 4
#         self.comp_size_mid = int(self.in_size / 2)
#         self.comp_size = int(self.comp_size_mid / 2)
#
#         self.dense = nn.Sequential(
#             nn.Linear(self.in_size, self.comp_size_mid),
#             nn.ReLU(),
#             nn.Linear(self.comp_size_mid, self.comp_size),
#             nn.ReLU(),
#         )
#
#         self.lstm = nn.LSTM(self.comp_size * 1 * 1, self.comp_size * 2 * 1, 1, batch_first=True, dropout=self.dropout)
#
#         self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab * 6)
#         self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab * 6)
#         self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab * 6)
#         self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab * 6)
#
#
#         self._initialize_weights()
#
#     def init_hidden(self, device, batch_size=256):
#         hidden = torch.zeros(1, batch_size, self.comp_size * 2, requires_grad=False).to(device)
#         cell = torch.zeros(1, batch_size, self.comp_size * 2, requires_grad=False).to(device)
#         return hidden, cell
#
#     def detach_hidden(self, hidden):
#         hidden, cell = hidden
#         hidden = hidden.detach()
#         cell = cell.detach()
#         return hidden, cell
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, x, hidden):
#         # Print the shape of the input tensor
#
#
#         batch_size, seq_len, feature, feature_length = x.shape
#
#         # Reshape to handle feature_length correctly for each feature
#         notes = x[:, :, 0, :].reshape(batch_size * seq_len, feature_length)
#         offsets = x[:, :, 1, :].reshape(batch_size * seq_len, feature_length)
#         durations = x[:, :, 2, :].reshape(batch_size * seq_len, feature_length)
#         velocities = x[:, :, 3, :].reshape(batch_size * seq_len, feature_length)
#
#
#         note_embedd = self.note_embedd(notes)
#         offset_embedd = self.offset_embedd(offsets)
#         duration_embedd = self.duration_embedd(durations)
#         velocity_embedd = self.velocity_embedd(velocities)
#
#         embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=2)
#         embeddings = embeddings.view(batch_size, seq_len, -1)
#         embeddings = F.dropout(embeddings, self.dropout)
#         embeddings = self.dense(embeddings)
#
#         lstm_out, hidden = self.lstm(embeddings, hidden)
#
#         lstm_out = lstm_out[:, -1, :]
#
#         # Fully connected layers for predictions
#         note_pred = self.fc_note(lstm_out).view(batch_size, 6, self.note_embedd.num_embeddings)
#         velocity_pred = self.fc_velocity(lstm_out).view(batch_size, 6, self.velocity_embedd.num_embeddings)
#         offset_pred = self.fc_offset(lstm_out).view(batch_size, 6, self.offset_embedd.num_embeddings)
#         duration_pred = self.fc_duration(lstm_out).view(batch_size, 6, self.duration_embedd.num_embeddings)
#
#
#         return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class SimpleLstm(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(SimpleLstm, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32

        # Embedding layers
        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)

        self.in_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.comp_size_mid = int(self.in_size / 2)
        self.comp_size = int(self.comp_size_mid / 2)

        self.dense = nn.Sequential(
            nn.BatchNorm1d(self.in_size),
            nn.Linear(self.in_size, self.comp_size_mid),
            nn.ReLU(),
        )

        self.conv = nn.Conv1d(self.comp_size_mid, self.in_size, kernel_size=1)

        self.lstm = nn.LSTM(self.comp_size, self.comp_size * 4, 2, batch_first=True, dropout=self.dropout)

        self.fc_note = nn.Linear(self.comp_size * 4, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.comp_size * 4, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.comp_size * 4, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.comp_size * 4, note_data.v_vocab)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=256):
        hidden = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):
        batch_size, seq_len, feature_dim = x.shape

        notes = x[:, :, 0]
        offsets = x[:, :, 1]
        durations = x[:, :, 2]
        velocities = x[:, :, 3]

        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        # Apply batch normalization to embeddings
        note_embedd = self.note_bn(note_embedd.permute(0, 2, 1)).permute(0, 2, 1)
        offset_embedd = self.offset_bn(offset_embedd.permute(0, 2, 1)).permute(0, 2, 1)
        duration_embedd = self.duration_bn(duration_embedd.permute(0, 2, 1)).permute(0, 2, 1)
        velocity_embedd = self.velocity_bn(velocity_embedd.permute(0, 2, 1)).permute(0, 2, 1)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=2)
        embeddings = F.dropout(embeddings, self.dropout)
        embeddings = self.dense(embeddings)

        lstm_out, hidden = self.lstm(embeddings, hidden)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.fc_note(lstm_out)
        offset_pred = self.fc_offset(lstm_out)
        duration_pred = self.fc_duration(lstm_out)
        velocity_pred = self.fc_velocity(lstm_out)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


# class LSTMNew1(nn.Module):
#     def __init__(self, note_data, dropout=0.3, bidirectional=False):
#         super(LSTMNew1, self).__init__()
#         self.dropout = dropout
#
#         self.bidirectional = bidirectional
#         self.note_data = note_data
#         self.embedding_size1 = 32
#         self.embedding_size2 = 32
#         self.embedding_size3 = 32
#         self.embedding_size4 = 32
#
#         self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
#         self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
#         self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
#         self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
#
#         self.in_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
#         self.comp_size_mid = int(self.in_size / 2)
#         self.comp_size = int(self.comp_size_mid / 2)
#
#         self.dense = nn.Sequential(
#             nn.Linear(self.in_size, self.comp_size_mid),
#             nn.ReLU(),
#             nn.Linear(self.comp_size_mid, self.comp_size),
#             nn.ReLU(),
#         )
#
#         self.lstm = nn.LSTM(self.comp_size, self.comp_size * 4, 2, batch_first=True, dropout=self.dropout)
#
#         self.fc_note = nn.Linear(self.comp_size * 4, note_data.n_vocab)
#         self.fc_offset = nn.Linear(self.comp_size * 4, note_data.o_vocab)
#         self.fc_duration = nn.Linear(self.comp_size * 4, note_data.d_vocab)
#         self.fc_velocity = nn.Linear(self.comp_size * 4, note_data.v_vocab)
#
#         self._initialize_weights()
#
#     def init_hidden(self, device, batch_size=256):
#         hidden = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
#         cell = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
#         return hidden, cell
#
#     def detach_hidden(self, hidden):
#         hidden, cell = hidden
#         hidden = hidden.detach()
#         cell = cell.detach()
#         return hidden, cell
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, x, hidden):
#         batch_size, seq_len, feature_dim = x.shape
#
#         notes = x[:, :, 0]
#         offsets = x[:, :, 1]
#         durations = x[:, :, 2]
#         velocities = x[:, :, 3]
#
#         note_embedd = self.note_embedd(notes)
#         offset_embedd = self.offset_embedd(offsets)
#         duration_embedd = self.duration_embedd(durations)
#         velocity_embedd = self.velocity_embedd(velocities)
#
#         embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=2)
#         embeddings = F.dropout(embeddings, self.dropout)
#         embeddings = self.dense(embeddings)
#
#         lstm_out, hidden = self.lstm(embeddings, hidden)
#         lstm_out = lstm_out[:, -1, :]
#
#         note_pred = self.fc_note(lstm_out)
#         offset_pred = self.fc_offset(lstm_out)
#         duration_pred = self.fc_duration(lstm_out)
#         velocity_pred = self.fc_velocity(lstm_out)
#
#         return note_pred, offset_pred, duration_pred, velocity_pred, hidden
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class LSTMNew1(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(LSTMNew1, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)

        self.in_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.comp_size = int(self.in_size / 2)
        # self.comp_size = int(self.comp_size_mid / 2)

        self.dense_emb = nn.Sequential(
            nn.Linear(self.in_size, self.comp_size),
            nn.ReLU(),
            # nn.Linear(self.comp_size_mid, self.comp_size),
            # nn.ReLU(),
        )
        # CNN layers with different kernel sizes
        self.cnn1 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=1, padding=0)
        self.cnn3 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=3, padding=0)
        self.cnn5 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=5, padding=0)
        self.cnn7 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=7, padding=0)
        # self.cnn8 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=8, padding=0)

        self.dense_cnn = nn.Sequential(
            nn.Linear(self.comp_size * 4 * 2, self.comp_size * 4),  # 128 channels * 5 CNNs
            nn.ReLU(),
            # nn.Linear(self.comp_size * 4, self.comp_size * 2),
            # nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.comp_size * 4, self.comp_size * 4, 2, batch_first=True, dropout=self.dropout)

        self.dense_lstm = nn.Sequential(
            nn.Linear(self.comp_size * 4, self.comp_size * 2),  # 128 channels * 5 CNNs
            nn.ReLU(),
            # nn.Linear(self.comp_size * 4, self.comp_size * 2),
            # nn.ReLU(),
        )

        self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=256):
        hidden = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.shape

        notes = x[:, :, 0]
        offsets = x[:, :, 1]
        durations = x[:, :, 2]
        velocities = x[:, :, 3]

        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=2)
        embeddings = F.dropout(embeddings, self.dropout)
        embeddings = self.dense_emb(embeddings)

        # CNN layers
        embeddings = embeddings.permute(0, 2, 1)  # Change to (batch_size, in_size, seq_len) for Conv1d

        cnn1_out = self.cnn1(embeddings)

        cnn3_out = F.pad(embeddings, (1, 1))
        cnn3_out = self.cnn3(cnn3_out)

        cnn5_out = F.pad(embeddings, (2, 2))
        cnn5_out = self.cnn5(cnn5_out)

        cnn7_out = F.pad(embeddings, (3, 3))
        cnn7_out = self.cnn7(cnn7_out)
        #
        # cnn8_out = F.pad(embeddings, (3, 4))
        # cnn8_out = self.cnn8(cnn8_out)

        cnn_out = torch.cat((cnn1_out, cnn3_out, cnn5_out, cnn7_out), dim=1)
        cnn_out = cnn_out.permute(0, 2, 1)  # Change back to (batch_size, seq_len, concatenated_channels)

        # Dense layer for dimensionality reduction
        cnn_out = self.dense_cnn(cnn_out)

        # Self-attention
        # attn_out = self.attention(reduced, reduced, reduced)

        lstm_out, hidden = self.lstm(cnn_out, hidden)
        lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dense_lstm(lstm_out)

        note_pred = self.fc_note(lstm_out)
        offset_pred = self.fc_offset(lstm_out)
        duration_pred = self.fc_duration(lstm_out)
        velocity_pred = self.fc_velocity(lstm_out)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class LSTMNew2(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(LSTMNew2, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)

        self.in_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.comp_size = int(self.in_size / 2)
        # self.comp_size = int(self.comp_size_mid / 2)

        self.dense_emb = nn.Sequential(
            nn.Linear(self.in_size, self.comp_size),
            nn.ReLU(),
            # nn.Linear(self.comp_size_mid, self.comp_size),
            # nn.ReLU(),
        )
        # CNN layers with different kernel sizes
        self.cnn1 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=1, padding=0)
        self.cnn3 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=3, padding=0)
        self.cnn5 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=5, padding=0)
        self.cnn7 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=7, padding=0)
        # self.cnn8 = nn.Conv1d(in_channels=self.comp_size, out_channels=self.comp_size * 2, kernel_size=8, padding=0)

        self.dense_cnn = nn.Sequential(
            nn.Linear(self.comp_size * 4 * 2, self.comp_size * 4),  # 128 channels * 5 CNNs
            nn.ReLU(),
            # nn.Linear(self.comp_size * 4, self.comp_size * 2),
            # nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.comp_size * 4, self.comp_size * 4, 2, batch_first=True, dropout=self.dropout)

        self.dense_lstm = nn.Sequential(
            nn.Linear(self.comp_size * 4, self.comp_size * 2),  # 128 channels * 5 CNNs
            nn.ReLU(),
            # nn.Linear(self.comp_size * 4, self.comp_size * 2),
            # nn.ReLU(),
        )

        self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=256):
        hidden = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):
        batch_size, seq_len, _ = x.shape

        notes = x[:, :, 0]
        offsets = x[:, :, 1]
        durations = x[:, :, 2]
        velocities = x[:, :, 3]

        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=2)
        embeddings = F.dropout(embeddings, self.dropout)
        embeddings = self.dense_emb(embeddings)

        # CNN layers
        embeddings = embeddings.permute(0, 2, 1)  # Change to (batch_size, in_size, seq_len) for Conv1d

        cnn1_out = self.cnn1(embeddings)

        cnn3_out = F.pad(embeddings, (1, 1))
        cnn3_out = self.cnn3(cnn3_out)

        cnn5_out = F.pad(embeddings, (2, 2))
        cnn5_out = self.cnn5(cnn5_out)

        cnn7_out = F.pad(embeddings, (3, 3))
        cnn7_out = self.cnn7(cnn7_out)
        #
        # cnn8_out = F.pad(embeddings, (3, 4))
        # cnn8_out = self.cnn8(cnn8_out)

        cnn_out = torch.cat((cnn1_out, cnn3_out, cnn5_out, cnn7_out), dim=1)
        cnn_out = cnn_out.permute(0, 2, 1)  # Change back to (batch_size, seq_len, concatenated_channels)

        # Dense layer for dimensionality reduction
        cnn_out = self.dense_cnn(cnn_out)

        # Self-attention
        # attn_out = self.attention(reduced, reduced, reduced)

        lstm_out, hidden = self.lstm(cnn_out, hidden)
        lstm_out = lstm_out[:, -1, :]

        lstm_out = self.dense_lstm(lstm_out)

        note_pred = self.fc_note(lstm_out)
        offset_pred = self.fc_offset(lstm_out)
        duration_pred = self.fc_duration(lstm_out)
        velocity_pred = self.fc_velocity(lstm_out)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class EmbConvLstPoly655(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(EmbConvLstPoly655, self).__init__()
        self.dropout = dropout

        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 16
        self.embedding_size2 = 16
        self.embedding_size3 = 16
        self.embedding_size4 = 16

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)
        self.comp_size = 64

        self.embAttn = SelfAttentionImpr(self.comp_size)

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 1, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=3, padding=0),
            nn.GroupNorm(32, self.comp_size * 2),
            nn.LeakyReLU(),
            nn.Conv1d(self.comp_size * 2, self.comp_size * 1, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=4, padding=0),
            nn.GroupNorm(32, self.comp_size * 2),
            nn.LeakyReLU(),
            nn.Conv1d(self.comp_size * 2, self.comp_size * 1, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(self.comp_size, self.comp_size * 2, kernel_size=8, padding=0),
            nn.GroupNorm(32, self.comp_size * 2),
            nn.LeakyReLU(),
            nn.Conv1d(self.comp_size * 2, self.comp_size * 1, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.fc_conv = nn.Sequential(
            nn.Linear(self.comp_size * 4, self.comp_size * 2),
            nn.LeakyReLU(),
            SelfAttentionImpr(self.comp_size * 2),
            Permute(0, 2, 1),
            nn.GroupNorm(32, self.comp_size * 2)
        )

        self.lstm = nn.LSTM(self.comp_size * 2 * 1, self.comp_size * 2 * 1, 2, dropout=dropout, batch_first=True)
        # self.lstm = LSTM_ARCH(self.comp_size * 2,self.comp_size * 4 * 1, self.comp_size * 4 * 1 , lstm=self._lstm, dropout_rate=dropout)

        self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab * 6)
        self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab * 6)
        self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab * 6)
        self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab * 6)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=96):
        hidden = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 2).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):

        notes = x[:, :, 0]
        offsets = x[:, :, 1]
        durations = x[:, :, 2]
        velocities = x[:, :, 3]

        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        # note_embedd = self.note_encode(note_embedd)
        # offset_embedd = self.offset_encode(offset_embedd)
        # duration_embedd = self.duration_encode(duration_embedd)
        # velocity_embedd = self.velocity_encode(velocity_embedd)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=2)
        # embeddings = embeddings.permute(0, 2, 1)  # shape [160, 32, 1024]
        embeddings = F.dropout(embeddings, self.dropout)
        #  embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.embAttn(embeddings)
        embeddings = embeddings.permute(0, 2, 1)

        # Apply the convolutions separately to the input embeddings
        conv_out1 = self.conv1(embeddings)  # no padding required for kernel_size=1

        padded_embeddings2 = F.pad(embeddings, (1, 1))  # Add 1 padding to both left and right for kernel_size=3
        conv_out2 = self.conv2(padded_embeddings2)

        padded_embeddings3 = F.pad(embeddings, (1, 2))  # Add 1 padding to left and 2 padding to right for kernel_size=4
        conv_out3 = self.conv3(padded_embeddings3)

        padded_embeddings4 = F.pad(embeddings, (3, 4))  # Add 3 padding to left and 4 padding to right for kernel_size=8
        conv_out4 = self.conv4(padded_embeddings4)

        conv_out = torch.cat((conv_out1, conv_out3, conv_out2, conv_out4), dim=1)
        conv_out = F.dropout(conv_out, self.dropout)
        conv_out = F.leaky_relu(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        conv_out = self.fc_conv(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        lstm_out, hidden = self.lstm(conv_out)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.dropout(lstm_out, self.dropout)

        note_pred = self.fc_note(lstm_out)
        offset_pred = self.fc_offset(lstm_out)
        duration_pred = self.fc_duration(lstm_out)
        velocity_pred = self.fc_velocity(lstm_out)
        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.unsqueeze(1).repeat(1, timestep, 1)
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.unsqueeze(0).unsqueeze(1).repeat(encoder_outputs.size(0), 1, 1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class CustomLSTMWithAttention(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(CustomLSTMWithAttention, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)

        self.in_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.comp_size_mid = int(self.in_size / 2)
        self.comp_size = int(self.comp_size_mid / 2)

        self.dense = nn.Sequential(
            nn.Linear(self.in_size, self.comp_size_mid),
            nn.ReLU(),
            nn.Linear(self.comp_size_mid, self.comp_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.comp_size, self.comp_size * 4, num_layers=2, batch_first=True, dropout=self.dropout,
                            bidirectional=self.bidirectional)

        self.attention = BahdanauAttention(self.comp_size * 4)

        self.fc_note = nn.Linear(self.comp_size * 8, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.comp_size * 8, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.comp_size * 8, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.comp_size * 8, note_data.v_vocab)

    def init_hidden(self, device, batch_size=256):
        num_directions = 2 if self.bidirectional else 1
        hidden = (torch.zeros(2 * num_directions, batch_size, self.comp_size * 4).to(device),
                  torch.zeros(2 * num_directions, batch_size, self.comp_size * 4).to(device))
        return hidden

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        batch_size, seq_len, feature_dim = x.shape

        notes = x[:, :, 0]
        offsets = x[:, :, 1]
        durations = x[:, :, 2]
        velocities = x[:, :, 3]

        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=2)
        embeddings = F.dropout(embeddings, self.dropout)
        embeddings = self.dense(embeddings)

        lstm_out, hidden = self.lstm(embeddings, hidden)

        attn_weights = self.attention(lstm_out[:, -1, :], lstm_out)
        context = attn_weights.bmm(lstm_out)
        context = context.squeeze(1)
        concat_output = torch.cat((lstm_out[:, -1, :], context), dim=1)

        note_pred = self.fc_note(concat_output)
        offset_pred = self.fc_offset(concat_output)
        duration_pred = self.fc_duration(concat_output)
        velocity_pred = self.fc_velocity(concat_output)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.hidden_size ** 0.5
        attn_weights = self.softmax(scores)
        context = torch.bmm(attn_weights, V)

        return context, attn_weights


class CustomLSTMWithSelfAttention(nn.Module):
    def __init__(self, note_data, dropout=0.5, bidirectional=False):
        super(CustomLSTMWithSelfAttention, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32

        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)

        self.in_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.comp_size_mid = int(self.in_size / 2)
        self.comp_size = self.comp_size_mid  # int(self.comp_size_mid / 2)

        self.dense = nn.Sequential(
            nn.Linear(self.in_size, self.comp_size_mid),
            nn.ReLU(),
            # nn.Linear(self.comp_size_mid, self.comp_size),
            # nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.comp_size, self.comp_size * 4, num_layers=2, batch_first=True, dropout=self.dropout,
                            bidirectional=self.bidirectional)

        # Adjust the input size of reduce_dim to match the concatenated output size
        # self.reduce_dim = nn.Linear(self.comp_size * 4 * 2, self.comp_size * 4)

        self.reduce_dim = nn.Linear(self.comp_size * 4, self.comp_size * 2)
        self.self_attention = SelfAttention(self.comp_size * 2)

        self.fc_note = nn.Linear(self.comp_size * 2, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.comp_size * 2, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.comp_size * 2, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.comp_size * 2, note_data.v_vocab)

    def init_hidden(self, device, batch_size=256):
        num_directions = 2 if self.bidirectional else 1
        hidden = (torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.comp_size * 4).to(device),
                  torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.comp_size * 4).to(device))
        return hidden

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        # batch_size, seq_len, feature_dim = x.shape

        notes = x[:, :, 0]
        offsets = x[:, :, 1]
        durations = x[:, :, 2]
        velocities = x[:, :, 3]

        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=2)
        embeddings = F.dropout(embeddings, self.dropout)
        embeddings = self.dense(embeddings)

        lstm_out, hidden = self.lstm(embeddings, hidden)

        # Only use lstm_out without hidden state concatenation
        lstm_out_last = lstm_out[:, -1, :]  # Output of the last time step

        # Apply self-attention
        lstm_out_last = lstm_out_last.unsqueeze(1)  # Adding sequence dimension for self-attention
        context, attn_weights = self.self_attention(lstm_out_last)
        context = context.squeeze(1)  # Removing sequence dimension after attention

        # # lstm_out_last = lstm_out[:, -1, :]
        # # hidden_last = hidden[0][-1]  # Using the last layer hidden state
        #
        # # alt
        # # lstm_out_last = lstm_out[:, -1, :]  # Output of the last time step
        # # hidden_last = hidden[0][-1, :, :]  # Hidden state of the last layer at the last time step
        #
        # # Concatenate lstm_out_last and hidden_last and reduce dimensionality
        # # concat_output = torch.cat((lstm_out_last, hidden_last), dim=1)
        # reduced_output = self.reduce_dim(lstm_out)
        #
        # # Apply self-attention
        # reduced_output = reduced_output.unsqueeze(1)  # Adding sequence dimension for self-attention
        # context, attn_weights = self.self_attention(reduced_output)
        # context = context.squeeze(1)  # Removing sequence dimension after attention

        note_pred = self.fc_note(context)
        offset_pred = self.fc_offset(context)
        duration_pred = self.fc_duration(context)
        velocity_pred = self.fc_velocity(context)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class SimpleLstm(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(SimpleLstm, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 32
        self.embedding_size2 = 32
        self.embedding_size3 = 32
        self.embedding_size4 = 32

        # Embedding layers
        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)

        # 2D CNN layers with kernel size 1 for each embedding
        self.note_conv = nn.Conv2d(1, 1, kernel_size=(1, self.embedding_size1), stride=1)
        self.offset_conv = nn.Conv2d(1, 1, kernel_size=(1, self.embedding_size2), stride=1)
        self.duration_conv = nn.Conv2d(1, 1, kernel_size=(1, self.embedding_size3), stride=1)
        self.velocity_conv = nn.Conv2d(1, 1, kernel_size=(1, self.embedding_size4), stride=1)

        # Batch normalization layer for concatenated embeddings
        self.concat_bn = nn.BatchNorm1d(
            self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4)

        # Additional CNN layer with kernel size 1 for concatenated features
        self.concat_conv = nn.Conv2d(1, 1, kernel_size=(
            1, self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4), stride=1)

        self.in_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.comp_size_mid = int(self.in_size / 2)
        self.comp_size = int(self.comp_size_mid / 2)

        self.dense = nn.Sequential(
            nn.Linear(self.in_size, self.comp_size_mid),
            nn.ReLU(),
            nn.Linear(self.comp_size_mid, self.comp_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.comp_size, self.comp_size * 4, 2, batch_first=True, dropout=self.dropout)

        self.fc_note = nn.Linear(self.comp_size * 4, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.comp_size * 4, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.comp_size * 4, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.comp_size * 4, note_data.v_vocab)

        self._initialize_weights()

    def init_hidden(self, device, batch_size=256):
        hidden = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
        cell = torch.zeros(2, batch_size, self.comp_size * 4, requires_grad=False).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):
        batch_size, seq_len, feature_dim = x.shape

        notes = x[:, :, 0].unsqueeze(1)  # (batch_size, 1, seq_len)
        offsets = x[:, :, 1].unsqueeze(1)
        durations = x[:, :, 2].unsqueeze(1)
        velocities = x[:, :, 3].unsqueeze(1)

        note_embedd = self.note_embedd(notes).permute(0, 3, 1, 2)  # (batch_size, embedding_size1, 1, seq_len)
        offset_embedd = self.offset_embedd(offsets).permute(0, 3, 1, 2)
        duration_embedd = self.duration_embedd(durations).permute(0, 3, 1, 2)
        velocity_embedd = self.velocity_embedd(velocities).permute(0, 3, 1, 2)

        # Apply 2D CNN with kernel size 1
        note_embedd = F.relu(self.note_conv(note_embedd)).squeeze(2)
        offset_embedd = F.relu(self.offset_conv(offset_embedd)).squeeze(2)
        duration_embedd = F.relu(self.duration_conv(duration_embedd)).squeeze(2)
        velocity_embedd = F.relu(self.velocity_conv(velocity_embedd)).squeeze(2)

        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=1)

        # Apply batch normalization to concatenated embeddings
        embeddings = self.concat_bn(embeddings.permute(0, 2, 1)).permute(0, 2, 1)

        # Apply additional CNN with kernel size 1 to the concatenated embeddings
        embeddings = F.relu(self.concat_conv(embeddings.unsqueeze(1))).squeeze(1)

        # Optionally, apply larger kernels to capture temporal dependencies (e.g., kernel_size=(3, 3))
        # embeddings = F.relu(self.larger_conv(embeddings.unsqueeze(1))).squeeze(1)

        embeddings = F.dropout(embeddings, self.dropout)
        embeddings = self.dense(embeddings)

        lstm_out, hidden = self.lstm(embeddings, hidden)
        lstm_out = lstm_out[:, -1, :]

        note_pred = self.fc_note(lstm_out)
        offset_pred = self.fc_offset(lstm_out)
        duration_pred = self.fc_duration(lstm_out)
        velocity_pred = self.fc_velocity(lstm_out)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class SimpleLstm(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(SimpleLstm, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 8
        self.embedding_size2 = 8
        self.embedding_size3 = 8
        self.embedding_size4 = 8

        # Embedding layers
        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)

        self.in_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.out_size = self.in_size * 2

        # Conv1D layers
        self.single_conv = nn.Conv1d(self.in_size, self.out_size, kernel_size=1)
        self.single_conv2 = nn.Conv1d(self.in_size, self.out_size, kernel_size=11, padding=5)

        self.multi_conv1 = nn.Sequential(
            nn.Conv1d(self.out_size, self.out_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.out_size, self.out_size * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_size * 2),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.out_size * 2, self.out_size, kernel_size=1, padding=0)
        )

        self.multi_conv2 = nn.Sequential(
            nn.Conv1d(self.out_size, self.out_size, kernel_size=3, dilation=3, stride=4, padding=3),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.out_size, self.out_size * 2, kernel_size=5, dilation=3, stride=4, padding=6),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_size * 2),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.out_size * 2, self.out_size, kernel_size=1, padding=0)
        )

        self.multi_conv3 = nn.Sequential(
            nn.Conv1d(self.out_size, self.out_size, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.out_size, self.out_size * 2, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_size * 2),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.out_size * 2, self.out_size, kernel_size=1, padding=0)
        )

        self.multi_conv4 = nn.Sequential(
            nn.Conv1d(self.out_size, self.out_size, kernel_size=7, dilation=3, stride=4, padding=9),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.out_size, self.out_size * 2, kernel_size=9, dilation=3, stride=4, padding=12),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_size * 2),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.out_size * 2, self.out_size, kernel_size=1, padding=0)
        )

        # LSTM layer
        self.lstm = nn.LSTM(self.out_size * 6, self.out_size * 6, num_layers=4, batch_first=True, dropout=self.dropout,
                            bidirectional=self.bidirectional)

        # Dense layer
        self.dense = nn.Sequential(
            nn.Linear(self.out_size * 6 * 2 if self.bidirectional else self.out_size * 6, self.out_size * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_size * 2, self.in_size),
            nn.ReLU()
        )

        # Fully connected layers for final output
        self.fc_note = nn.Linear(self.in_size, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.in_size, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.in_size, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.in_size, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=256):
        num_directions = 2 if self.bidirectional else 1
        hidden = torch.zeros(4 * num_directions, batch_size, self.out_size * 6, requires_grad=False).to(device)
        cell = torch.zeros(4 * num_directions, batch_size, self.out_size * 6, requires_grad=False).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        batch_size, seq_len, feature_dim = x.shape

        notes = x[:, :, 0]
        offsets = x[:, :, 1]
        durations = x[:, :, 2]
        velocities = x[:, :, 3]

        # Embeddings
        note_embedd = self.note_embedd(notes).permute(0, 2, 1)  # (batch_size, embedding_size1, seq_len)
        offset_embedd = self.offset_embedd(offsets).permute(0, 2, 1)
        duration_embedd = self.duration_embedd(durations).permute(0, 2, 1)
        velocity_embedd = self.velocity_embedd(velocities).permute(0, 2, 1)

        # Concatenate embeddings
        embeddings = torch.cat((note_embedd, offset_embedd, duration_embedd, velocity_embedd), dim=1)

        # Apply single Conv1D layers
        single_conv_out1 = F.relu(self.single_conv(embeddings))
        single_conv_out2 = F.relu(self.single_conv2(embeddings))

        # Apply multi Conv1D layers to single_conv_out1
        multi_conv_out1 = self.multi_conv1(single_conv_out1)
        multi_conv_out2 = self.multi_conv2(single_conv_out1)
        multi_conv_out3 = self.multi_conv3(single_conv_out1)
        multi_conv_out4 = self.multi_conv4(single_conv_out1)

        # Ensure the output dimensions match for concatenation
        max_seq_len = min(single_conv_out1.size(2), single_conv_out2.size(2),
                          multi_conv_out1.size(2), multi_conv_out2.size(2),
                          multi_conv_out3.size(2), multi_conv_out4.size(2))

        single_conv_out1 = single_conv_out1[:, :, :max_seq_len]
        single_conv_out2 = single_conv_out2[:, :, :max_seq_len]
        multi_conv_out1 = multi_conv_out1[:, :, :max_seq_len]
        multi_conv_out2 = multi_conv_out2[:, :, :max_seq_len]
        multi_conv_out3 = multi_conv_out3[:, :, :max_seq_len]
        multi_conv_out4 = multi_conv_out4[:, :, :max_seq_len]

        # Concatenate all Conv1D outputs
        combined_conv_out = torch.cat((single_conv_out1, single_conv_out2,
                                       multi_conv_out1, multi_conv_out2,
                                       multi_conv_out3, multi_conv_out4), dim=1)

        # Add embedding concatenation to combined Conv1D outputs

        # LSTM layer
        lstm_out, hidden = self.lstm(combined_conv_out.permute(0, 2, 1), hidden)
        lstm_out = lstm_out[:, -1, :]  # Use the output of the last time step

        # Dense layer
        dense_out = self.dense(lstm_out)

        # Fully connected layers
        note_pred = self.fc_note(dense_out)
        offset_pred = self.fc_offset(dense_out)
        duration_pred = self.fc_duration(dense_out)
        velocity_pred = self.fc_velocity(dense_out)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden


class AdvancedLstmModel(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(AdvancedLstmModel, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.note_data = note_data
        self.embedding_size1 = 8
        self.embedding_size2 = 8
        self.embedding_size3 = 8
        self.embedding_size4 = 8

        # Embedding layers
        self.note_embedd = nn.Embedding(note_data.n_vocab, self.embedding_size1, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, self.embedding_size2, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, self.embedding_size3, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, self.embedding_size4, padding_idx=0)

        self.in_size = self.embedding_size1 + self.embedding_size2 + self.embedding_size3 + self.embedding_size4
        self.out_size = self.in_size * 2

        # Convolutional layers
        self.single_conv = nn.Conv1d(self.in_size, self.out_size, kernel_size=1)
        self.single_conv2 = nn.Conv1d(self.in_size, self.out_size, kernel_size=11, padding=5)

        # Pointwise convolution to match dimensions
        self.match_conv = nn.Conv1d(self.in_size, self.out_size * 2, kernel_size=1)

        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.out_size * 8, num_heads=4, dropout=self.dropout)

        # Multi-Layer LSTM
        self.lstm = nn.LSTM(self.out_size * 8, self.out_size * 8, num_layers=4, batch_first=True, dropout=self.dropout, bidirectional=False)

        self.dense = nn.Sequential(
            nn.Linear(self.out_size * 8, self.out_size * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_size * 4, self.out_size * 2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.out_size * 2, self.in_size),
            nn.ReLU(),
        )

        # Dense layers for final output
        self.fc_note = nn.Linear(self.in_size, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.in_size, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.in_size, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.in_size, note_data.v_vocab)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def init_hidden(self, device, batch_size=256):
        hidden = torch.zeros(4, batch_size, self.out_size * 8, requires_grad=False).to(device)
        cell = torch.zeros(4, batch_size, self.out_size * 8, requires_grad=False).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden=None):
        batch_size, seq_len, feature_dim = x.shape

        # Split x into the four features
        note = x[:, :, 0]
        offset = x[:, :, 1]
        duration = x[:, :, 2]
        velocity = x[:, :, 3]

        note_emb = self.note_embedd(note)
        offset_emb = self.offset_embedd(offset)
        duration_emb = self.duration_embedd(duration)
        velocity_emb = self.velocity_embedd(velocity)

        # Concatenate embeddings
        x = torch.cat((note_emb, offset_emb, duration_emb, velocity_emb), dim=-1).transpose(1, 2)

        # Pass through the initial single convolution
        single_conv_out = self.single_conv(x)
        single_conv2_out = self.single_conv2(x)

        # Concatenate convolution outputs
        conv_out = torch.cat((single_conv_out, single_conv2_out), dim=1)

        # Adjust dimensions of x to match conv_out
        x_matched = self.match_conv(x)

        # Add original embedding to the concatenation (residual connection)
        embed_out = x_matched + conv_out

        # Prepare for multi-head attention
        embed_out = embed_out.transpose(0, 1)  # Required for multi-head attention

        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(embed_out, embed_out, embed_out)

        # Pass through LSTM
        lstm_out, hidden = self.lstm(attn_output.transpose(0, 1), hidden)

        # Take the last output of the LSTM
        lstm_out = lstm_out[:, -1, :]

        # Pass through dense layers
        lstm_out = self.dense(lstm_out)

        # Final output layers
        note_out = self.fc_note(lstm_out)
        offset_out = self.fc_offset(lstm_out)
        duration_out = self.fc_duration(lstm_out)
        velocity_out = self.fc_velocity(lstm_out)

        return note_out, offset_out, duration_out, velocity_out, hidden



class SimpleLstm(nn.Module):
    def __init__(self, note_data, dropout=0.3, bidirectional=False):
        super(SimpleLstm, self).__init__()
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.note_data = note_data
        self.hidden_size = 128

        # Embedding layers
        self.note_embedd = nn.Embedding(note_data.n_vocab, 32, padding_idx=0)
        self.offset_embedd = nn.Embedding(note_data.o_vocab, 32, padding_idx=0)
        self.duration_embedd = nn.Embedding(note_data.d_vocab, 32, padding_idx=0)
        self.velocity_embedd = nn.Embedding(note_data.v_vocab, 32, padding_idx=0)

        # LSTMs for each feature
        self.lstm_note = nn.LSTM(32, self.hidden_size, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        self.lstm_offset = nn.LSTM(32, self.hidden_size, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        self.lstm_duration = nn.LSTM(32, self.hidden_size, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)
        self.lstm_velocity = nn.LSTM(32, self.hidden_size, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        # Final LSTM after concatenation
        self.final_lstm = nn.LSTM(self.hidden_size * 4, self.hidden_size, num_layers=2, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

        # Fully connected layers for final output
        self.fc_note = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, note_data.n_vocab)
        self.fc_offset = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, note_data.o_vocab)
        self.fc_duration = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, note_data.d_vocab)
        self.fc_velocity = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, note_data.v_vocab)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=256):
        num_directions = 2 if self.bidirectional else 1
        hidden = (torch.zeros(2 * num_directions, batch_size, self.hidden_size).to(device),
                  torch.zeros(2 * num_directions, batch_size, self.hidden_size).to(device))
        return hidden


    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell


    def forward(self, x, hidden):
        batch_size, seq_len, feature_dim = x.shape

        notes = x[:, :, 0]
        offsets = x[:, :, 1]
        durations = x[:, :, 2]
        velocities = x[:, :, 3]

        # Embeddings
        note_embedd = self.note_embedd(notes)
        offset_embedd = self.offset_embedd(offsets)
        duration_embedd = self.duration_embedd(durations)
        velocity_embedd = self.velocity_embedd(velocities)

        # LSTM for each feature
        _, (hidden_note, _) = self.lstm_note(note_embedd)
        _, (hidden_offset, _) = self.lstm_offset(offset_embedd)
        _, (hidden_duration, _) = self.lstm_duration(duration_embedd)
        _, (hidden_velocity, _) = self.lstm_velocity(velocity_embedd)

        # Concatenate hidden states
        hidden_concat = torch.cat((hidden_note[-1], hidden_offset[-1], hidden_duration[-1], hidden_velocity[-1]), dim=1).unsqueeze(1)

        # Final LSTM
        lstm_out, hidden_final = self.final_lstm(hidden_concat)

        # Use the output of the last time step
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layers
        note_pred = self.fc_note(lstm_out)
        offset_pred = self.fc_offset(lstm_out)
        duration_pred = self.fc_duration(lstm_out)
        velocity_pred = self.fc_velocity(lstm_out)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden_final



# class CRNNModel(nn.Module):
#     def __init__(self, note_data, hidden_dim, num_layers):
#         super(CRNNModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.embedding_note = nn.Embedding(note_data.n_vocab, hidden_dim, padding_idx=0)
#         self.embedding_offset = nn.Embedding(note_data.o_vocab, hidden_dim, padding_idx=0)
#         self.embedding_duration = nn.Embedding(note_data.d_vocab, hidden_dim, padding_idx=0)
#         self.embedding_velocity = nn.Embedding(note_data.v_vocab, hidden_dim, padding_idx=0)
#
#         self.conv1 = nn.Conv1d(hidden_dim * 4, hidden_dim * 8, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(hidden_dim * 8, hidden_dim * 16, kernel_size=3, padding=1)
#
#         self.lstm = nn.LSTM(hidden_dim * 16, hidden_dim, num_layers, batch_first=True)
#
#         self.fc_note = nn.Linear(hidden_dim, note_data.n_vocab)
#         self.fc_offset = nn.Linear(hidden_dim, note_data.o_vocab)
#         self.fc_duration = nn.Linear(hidden_dim, note_data.d_vocab)
#         self.fc_velocity = nn.Linear(hidden_dim, note_data.v_vocab)
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def init_hidden(self, device, batch_size=256):
#         num_directions = 1 # 2 if self.bidirectional else 1
#         hidden = (torch.zeros(2 * num_directions, batch_size, self.hidden_dim * 16).to(device),
#                   torch.zeros(2 * num_directions, batch_size, self.hidden_dim * 16).to(device))
#         return hidden
#
#
#     def detach_hidden(self, hidden):
#         hidden, cell = hidden
#         hidden = hidden.detach()
#         cell = cell.detach()
#         return hidden, cell
#
#     def forward(self, x, hidden):
#         batch_size, seq_len, feature_dim = x.shape
#
#         notes = self.embedding_note(x[:, :, 0])
#         offsets = self.embedding_offset(x[:, :, 1])
#         durations = self.embedding_duration(x[:, :, 2])
#         velocities = self.embedding_velocity(x[:, :, 3])
#
#         combined = torch.cat((notes, offsets, durations, velocities), dim=2)
#         combined = combined.permute(0, 2, 1)  # For Conv1d, input should be (batch_size, feature_dim, seq_len)
#
#         conv_out = F.relu(self.conv1(combined))
#         conv_out = F.relu(self.conv2(conv_out))
#
#         conv_out = conv_out.permute(0, 2, 1)  # For LSTM, input should be (batch_size, seq_len, feature_dim)
#         lstm_out, hidden = self.lstm(conv_out, hidden)
#
#         self.fc_note = nn.Linear(hidden_dim, note_data.n_vocab)
#         self.fc_offset = nn.Linear(hidden_dim, note_data.o_vocab)
#         self.fc_duration = nn.Linear(hidden_dim, note_data.d_vocab)
#         self.fc_velocity = nn.Linear(hidden_dim, note_data.v_vocab)
#
#         return note_pred, offset_pred, duration_pred, velocity_pred, hidden

class MultiInputLSTMWithAttention(nn.Module):
    def __init__(self, note_data, hidden_dim, num_layers):
        super(MultiInputLSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding_note = nn.Embedding(note_data.n_vocab, hidden_dim, padding_idx=0)
        self.embedding_offset = nn.Embedding(note_data.o_vocab, hidden_dim, padding_idx=0)
        self.embedding_duration = nn.Embedding(note_data.d_vocab, hidden_dim, padding_idx=0)
        self.embedding_velocity = nn.Embedding(note_data.v_vocab, hidden_dim, padding_idx=0)

        self.lstm_note = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_offset = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_duration = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_velocity = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        self.attention = nn.MultiheadAttention(hidden_dim * 4, num_heads=4, batch_first=True)

        self.fc_note = nn.Linear(hidden_dim * 4, note_data.n_vocab)
        self.fc_offset = nn.Linear(hidden_dim * 4, note_data.o_vocab)
        self.fc_duration = nn.Linear(hidden_dim * 4, note_data.d_vocab)
        self.fc_velocity = nn.Linear(hidden_dim * 4, note_data.v_vocab)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, device, batch_size=256):
        num_directions = 1  # 2 if bidirectional, here it's not
        hidden = (torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_dim).to(device))
        return hidden

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, x, hidden):
        batch_size, seq_len, feature_dim = x.shape

        notes = self.embedding_note(x[:, :, 0])
        offsets = self.embedding_offset(x[:, :, 1])
        durations = self.embedding_duration(x[:, :, 2])
        velocities = self.embedding_velocity(x[:, :, 3])

        lstm_out_note, hidden_note = self.lstm_note(notes, hidden)
        lstm_out_offset, hidden_offset = self.lstm_offset(offsets, hidden)
        lstm_out_duration, hidden_duration = self.lstm_duration(durations, hidden)
        lstm_out_velocity, hidden_velocity = self.lstm_velocity(velocities, hidden)

        combined = torch.cat((lstm_out_note, lstm_out_offset, lstm_out_duration, lstm_out_velocity), dim=2)
        attn_output, attn_weights = self.attention(combined, combined, combined)

        # Ensure the output shape matches (batch_size * seq_len, vocab_size)
        note_pred = self.fc_note(attn_output).reshape(batch_size * seq_len, -1)
        offset_pred = self.fc_offset(attn_output).reshape(batch_size * seq_len, -1)
        duration_pred = self.fc_duration(attn_output).reshape(batch_size * seq_len, -1)
        velocity_pred = self.fc_velocity(attn_output).reshape(batch_size * seq_len, -1)

        return note_pred, offset_pred, duration_pred, velocity_pred, hidden