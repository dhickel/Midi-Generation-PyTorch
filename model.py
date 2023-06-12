import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        return output_note, output_offset, output_duration, output_velocity \




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


class SelfAttentionNorm(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionNorm, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        weights = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1]), dim=-1)
        return self.layernorm(weights @ v)



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



        self.conv6 = nn.Conv1d(self.e_size, self.e_size * 2,  kernel_size=16, padding=0)


        self.attention2 = SelfAttention(self.e_size * 6 * 2)

        #self.ln1 = nn.LayerNorm(self.e_size * 2 * 5)  # Change size as needed

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

        padded_embeddings6 = F.pad(embeddings, (7, 8))  # Add 7 padding to left and 8 padding to right for kernel_size=16
        conv_out61 = self.conv6(padded_embeddings6)



        # conv_out6 = (conv_out6 + conv_out61) / 2.0


        # Concatenate along the channel dimension
        conv_out = torch.cat((conv_out1, conv_out2 , conv_out4, conv_out5, conv_out6, conv_out61), dim=1)



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