import glob
import pickle
import random

import numpy as np
import torch
from music21 import converter, instrument, note, chord, stream, duration, pitch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from model import *


class NoteData:
    def __init__(self):
        self.note_table = []
        self.duration_table = []
        self.offset_table = []
        self.note_max = -1
        self.offset_max = -1
        self.duration_max = -1
        self.training_notes = []
        self.n_vocab = 0
        self.o_vocab = 0
        self.d_vocab = 0
        self.rand = [0.25, 0.5, 0.75, 1]

    @staticmethod
    def contains(lst, item):
        for i in range(0, len(lst)):
            if lst[i] == item:
                return i

    def calc_max(self):
        self.note_max = max(self.note_table)
        self.offset_max = max(self.offset_table)
        self.duration_max = max(self.duration_table)
        self.n_vocab = len(self.note_table)
        self.o_vocab = len(self.offset_table)
        self.d_vocab = len(self.duration_table)

    def add_note_if_absent(self, note):
        idx = self.contains(self.note_table, note)
        if idx is None:
            self.note_table.append(note)
            return len(self.note_table) - 1
        else:
            return idx

    def add_durr_if_absent(self, duration):
        idx = self.contains(self.duration_table, duration)
        if idx is None:
            self.duration_table.append(duration)
            return len(self.duration_table) - 1
        else:
            return idx

    def add_offs_if_absent(self, offset):
        off = round(offset, 3)
        if off > 4:
            off = .25
        idx = self.contains(self.offset_table, off)
        if idx is None:
            self.offset_table.append(off)
            return len(self.offset_table) - 1
        else:
            return idx

    def get_random_off(self):
        if len(self.offset_table) < 4:
            return 0.25
        else:
            return self.rand[random.randint(0, len(self.rand) - 1)]


class MusicDataset(Dataset):
    def __init__(self, network_data):
        self.network_input = network_data.input
        self.network_output_notes = network_data.output_notes
        self.network_output_offsets = network_data.output_offsets
        self.network_output_durations = network_data.output_durations

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], (
            self.network_output_notes[idx], self.network_output_offsets[idx], self.network_output_durations[idx])


class NetworkData:
    def __init__(self, network_input, network_output_notes, network_output_offsets, network_output_durations):
        self.input = network_input
        self.output_notes = network_output_notes
        self.output_offsets = network_output_offsets
        self.output_durations = network_output_durations


def get_notes(directory):
    data = NoteData()

    for file in glob.glob(f'{directory}*.mid'):
        print("Parsing: ", file)
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"Warning: could not parse {file}. Skipping. Error: {e}")
            continue

        try:  # file has instrument parts
            instruments = instrument.partitionByInstrument(midi)
            notes_to_parse = instruments.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notesAndRests

        prev_event_end = data.get_random_off()
        for event in notes_to_parse:

            if isinstance(event, note.Note) or isinstance(event, chord.Chord):
                offset = event.offset - prev_event_end  # Calc gap
                inst = event.activeSite.getInstrument()
                if inst and inst.midiProgram is not None and inst.midiProgram >= 110:  # Filter out percussion
                    continue
                note_val = None
                durr_val = None
                if isinstance(event, note.Note):
                    note_val = str(event.pitch.nameWithOctave)
                    durr_val = float(event.duration.quarterLength)
                    if offset < 0:
                        offset = 0
                elif isinstance(event, chord.Chord):
                    note_val = '.'.join(str(p.nameWithOctave) for p in event.pitches)
                    durr_val = float(event.duration.quarterLength)
                    if offset <= 0:
                        pass
                        offset = data.get_random_off()

                if note_val is not None and durr_val is not None:
                    note_val = data.add_note_if_absent(note_val)
                    durr_val = data.add_durr_if_absent(durr_val)
                    offset = data.add_offs_if_absent(offset)
                    data.training_notes.append((note_val, offset, durr_val))
                prev_event_end = event.offset

        data.calc_max()
    print(f'Notess{data.note_table}')
    print(f'Offsets{data.offset_table}')
    print(f'Durations{data.duration_table}')
    return data


#
# def prepare_sequences(note_data):
#     sequence_length = 32
#
#     network_input = []
#     network_output_notes = []
#     network_output_offsets = []
#     network_output_durations = []
#
#     # create input sequences and the corresponding outputs
#     for i in range(0, len(note_data.training_notes) - sequence_length, 1):
#         sequence_in = note_data.training_notes[i:i + sequence_length]
#         sequence_out = note_data.training_notes[i + sequence_length]
#         network_input.append([[x[0], x[1], x[2]] for x in sequence_in])
#         network_output_notes.append(sequence_out[0])
#         network_output_offsets.append(sequence_out[1])
#         network_output_durations.append(sequence_out[2])
#
#     # network_input = torch.Tensor(network_input)
#     # network_output_notes = torch.LongTensor(network_output_notes)
#     # network_output_offsets = torch.LongTensor(network_output_offsets)
#     # network_output_durations = torch.LongTensor(network_output_durations)
#
#     network_input = torch.tensor(network_input, dtype=torch.float16)
#     network_output_notes = torch.tensor(network_output_notes, dtype=torch.long)
#     network_output_offsets = torch.tensor(network_output_offsets, dtype=torch.long)
#     network_output_durations = torch.tensor(network_output_durations, dtype=torch.long)
#
#     return NetworkData(network_input, network_output_notes, network_output_offsets, network_output_durations)


def prepare_sequences(note_data):
    sequence_length = 64

    network_input = []
    network_output_notes = []
    network_output_offsets = []
    network_output_durations = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(note_data.training_notes) - sequence_length, 1):
        sequence_in = note_data.training_notes[i:i + sequence_length]
        sequence_out = note_data.training_notes[i + sequence_length]
        network_input.append([[x[0], x[1], x[2]] for x in sequence_in])
        network_output_notes.append(sequence_out[0])
        network_output_offsets.append(sequence_out[1])
        network_output_durations.append(sequence_out[2])

    network_input = torch.tensor(network_input, dtype=torch.float16)

    # Shape for cross_entropy: (N) where N is batch size.
    network_output_notes = torch.tensor(network_output_notes, dtype=torch.long).view(-1)
    network_output_offsets = torch.tensor(network_output_offsets, dtype=torch.long).view(-1)
    network_output_durations = torch.tensor(network_output_durations, dtype=torch.long).view(-1)

    return NetworkData(network_input, network_output_notes, network_output_offsets, network_output_durations)


# class ConvLSTM(nn.Module):
#
#     def __init__(self, note_data):
#         super(ConvLSTM, self).__init__()
#
#         self.cnn = nn.Sequential(
#             nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         )
#
#         self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, dropout=0.3)
#         self.fc_intermediate = nn.Linear(256, 512)  # Additional fully connected layer
#         self.fc_note = nn.Linear(512, note_data.n_vocab)
#         self.fc_offset = nn.Linear(512, note_data.o_vocab)
#         self.fc_duration = nn.Linear(512, note_data.d_vocab)
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         batch_size, seq_length, _ = x.size()
#
#         # Treat each feature as a separate channel
#         cnn_out = self.cnn(x.view(batch_size * seq_length, 3, -1))  # (batch_size * seq_length, num_channels, 1)
#
#         # Now, reshape it in a way to retain the sequence length for LSTM
#         cnn_out = cnn_out.view(batch_size, seq_length, -1)
#
#         # lstm_out, _ = self.lstm(cnn_out)
#         # lstm_out = lstm_out[:, -1, :]  # Only take the output from the final timestep
#         # lstm_out = self.fc_intermediate(lstm_out)  # Pass through additional fully connected layer
#         # output_note = self.fc_note(lstm_out)
#         # output_offset = self.fc_offset(lstm_out)
#         # output_duration = self.fc_duration(lstm_out)
#
#         lstm_out = self.lstm(cnn_out)[0][:, -1, :]
#         lstm_out = torch.relu(self.fc_intermediate(lstm_out))  # Apply ReLU activation function
#         output_note = self.fc_note(lstm_out)
#         output_offset = self.fc_offset(lstm_out)
#         output_duration = self.fc_duration(lstm_out)
#
#         return output_note, output_offset, output_duration

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)






class ConvLSTM(nn.Module):

    def __init__(self, note_data, dropout=0.2):
        super(ConvLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout after first Conv1D and ReLU
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)  # Dropout after second Conv1D and ReLU
        )

        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, dropout=dropout)
        self.fc_intermediate = nn.Linear(256, 512)  # Additional fully connected layer
        self.dropout = nn.Dropout(dropout)  # Define dropout for use after the FC layers
        self.fc_note = nn.Linear(512, note_data.n_vocab)
        self.fc_offset = nn.Linear(512, note_data.o_vocab)
        self.fc_duration = nn.Linear(512, note_data.d_vocab)

        self._initialize_weights()

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Treat each feature as a separate channel
        cnn_out = self.cnn(x.view(batch_size * seq_length, 3, -1))

        # Now, reshape it in a way to retain the sequence length for LSTM
        cnn_out = cnn_out.view(batch_size, seq_length, -1)

        lstm_out = self.lstm(cnn_out)[0][:, -1, :]
        lstm_out = torch.relu(self.fc_intermediate(lstm_out))  # Apply ReLU activation function
        lstm_out = self.dropout(lstm_out)  # Apply dropout after ReLU and intermediate FC layer

        output_note = self.fc_note(lstm_out)
        output_offset = self.fc_offset(lstm_out)
        output_duration = self.fc_duration(lstm_out)

        return output_note, output_offset, output_duration

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



#
# class ConvLSTM(nn.Module):
#     def __init__(self, note_data, input_dim=3, conv_feature_dim=64, hidden_dim=256, num_layers=2):
#         super(ConvLSTM, self).__init__()
#
#         # Conv1D Layer
#         self.conv1 = nn.Conv1d(input_dim, conv_feature_dim, kernel_size=3, stride=1, padding=1)
#
#         # LSTM Layer
#         self.lstm = nn.LSTM(conv_feature_dim, hidden_dim, num_layers, batch_first=True)
#
#         # Fully connected layer
#         self.fc_note = nn.Linear(hidden_dim, note_data.n_vocab)
#         self.fc_offset = nn.Linear(hidden_dim, note_data.o_vocab)
#         self.fc_duration = nn.Linear(hidden_dim, note_data.d_vocab)
#
#     def forward(self, x):
#         # Input shape (batch, sequence, features)
#         x = x.transpose(1, 2)  # switch sequence and features for Conv1D
#         x = self.conv1(x)
#         x = x.transpose(1, 2)  # switch back
#
#         # Passing in the output of the convolutional layer to the LSTM
#         out, (hn, cn) = self.lstm(x)
#
#         # only need the final time step output
#         out = out[:, -1, :]
#
#         # Calculating the output of the Linear layer for each output
#         out_note = self.fc_note(out)
#         out_offset = self.fc_offset(out)
#         out_duration = self.fc_duration(out)
#         return out_note, out_offset, out_duration


def generate_seed_from_int(seed_int, seq_length, note_data):
    # Create a random number generator with the provided seed
    rng = np.random.default_rng(seed_int)

    # Generate random indices within the range of each vocabulary
    note_indices = rng.integers(note_data.n_vocab, size=seq_length)
    offset_indices = rng.integers(note_data.o_vocab, size=seq_length)
    duration_indices = rng.integers(note_data.d_vocab, size=seq_length)

    # Stack the indices into a single sequence and reshape it to the required shape
    seed_sequence = np.vstack([note_indices, offset_indices, duration_indices])
    seed_sequence = seed_sequence.T.reshape(1, seq_length, 3)

    # Convert to a PyTorch tensor
    seed_sequence = torch.tensor(seed_sequence, dtype=torch.float16)

    return seed_sequence


def generate_notes2(model, note_data, network_data, device, seq_length=200, seed_sequence=None, temperature=1.0,):
    """ Generate notes from the neural network based on a sequence of notes """
    model.eval()

    # If a seed sequence is provided, use it; otherwise, choose one randomly
    if seed_sequence is None:
        start = np.random.randint(0, len(network_data.input) - 1)
        pattern = network_data.input[start].unsqueeze(0).to(device)
    else:
        pattern = seed_sequence

    prediction_output = []

    for note_index in range(seq_length):
        with torch.no_grad():
            prediction_note, prediction_offset, prediction_duration = model(pattern)

        note_index = torch.multinomial(F.softmax(prediction_note / temperature, dim=1), 1)
        offset_index = torch.multinomial(F.softmax(prediction_offset / temperature, dim=1), 1)
        duration_index = torch.multinomial(F.softmax(prediction_duration / temperature, dim=1), 1)


        result = (note_data.note_table[note_index[0, 0].item()],
                  note_data.offset_table[offset_index[0, 0].item()],
                  note_data.duration_table[duration_index[0, 0].item()])
        prediction_output.append(result)

        next_input = torch.tensor([[[note_index.item(), offset_index.item(), duration_index.item()]]],
                                  dtype=torch.float16).to(device)

        pattern = torch.cat((pattern[:, 1:, :], next_input), dim=1)

    return prediction_output


def train(model, train_loader, criterion, optimizer, device, note_data):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, (targets_note, targets_offset, targets_duration) in train_loader:
        inputs, targets_note, targets_offset, targets_duration = inputs.to(device), targets_note.to(
            device), targets_offset.to(device), targets_duration.to(device)

        # Forward pass
        output_note, output_offset, output_duration = model(inputs)

        # Calculate loss
        loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long())
        loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long())
        loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long())

        # loss_note = criterion(output_note, targets_note.long())
        # loss_offset = criterion(output_offset, targets_offset.long())
        # loss_duration = criterion(output_duration, targets_duration.long())

        loss = loss_note + loss_offset + loss_duration

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(train_loader.dataset)


def validate(model, valid_loader, criterion, device, note_data):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0

    with torch.no_grad():
        for inputs, (targets_note, targets_offset, targets_duration) in valid_loader:
            inputs, targets_note, targets_offset, targets_duration = inputs.to(device), targets_note.to(
                device), targets_offset.to(device), targets_duration.to(device)

            # Forward pass
            output_note, output_offset, output_duration = model(inputs)

            # Calculate loss
            loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long())
            loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long())
            loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long())

            loss = loss_note + loss_offset + loss_duration

            running_loss += loss.item() * inputs.size(0)

    return running_loss / len(valid_loader.dataset)


def generate_midi(model, note_data, network_data, output_file='output.mid'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    prediction_output = generate_notes(model, note_data, network_data, device)
    create_midi(prediction_output, output_file)


# def generate_notes(model, note_data, network_data, device):
#     """ Generate notes from the neural network based on a sequence of notes """
#     model.eval()
#     start = np.random.randint(0, len(network_data.input) - 1)
#
#     pattern = network_data.input[start].unsqueeze(0).to(device)
#
#     prediction_output = []
#
#     for note_index in range(200):
#         with torch.no_grad():
#             prediction_note, prediction_offset, prediction_duration = model(pattern)
#
#         note_index = torch.multinomial(F.softmax(prediction_note, dim=2), 1)
#         offset_index = torch.multinomial(F.softmax(prediction_offset, dim=2), 1)
#         duration_index = torch.multinomial(F.softmax(prediction_duration, dim=2), 1)
#
#         result = (note_data.note_table[note_index[0, 0].item()],
#                   note_data.offset_table[offset_index[0, 0].item()],
#                   note_data.duration_table[duration_index[0, 0].item()])
#         prediction_output.append(result)
#
#         next_input = torch.tensor([[[note_index.item(), offset_index.item(), duration_index.item()]]],
#                                   dtype=torch.float16).to(device)
#
#         pattern = torch.cat((pattern[:, 1:, :], next_input), dim=1)
#
#     return prediction_output


def generate_notes(model, note_data, network_data, device):
    """ Generate notes from the neural network based on a sequence of notes """
    model.eval()
    start = np.random.randint(0, len(network_data.input) - 1)

    pattern = network_data.input[start].unsqueeze(0).to(device)

    prediction_output = []

    for note_index in range(200):
        with torch.no_grad():
            prediction_note, prediction_offset, prediction_duration = model(pattern)

        note_index = torch.multinomial(F.softmax(prediction_note, dim=1), 1)
        offset_index = torch.multinomial(F.softmax(prediction_offset, dim=1), 1)
        duration_index = torch.multinomial(F.softmax(prediction_duration, dim=1), 1)

        result = (note_data.note_table[note_index[0, 0].item()],
                   note_data.offset_table[offset_index[0, 0].item()],
                  note_data.duration_table[duration_index[0, 0].item()])
        prediction_output.append(result)

        next_input = torch.tensor([[[note_index.item(), offset_index.item(), duration_index.item()]]],
                                  dtype=torch.float16).to(device)

        pattern = torch.cat((pattern[:, 1:, :], next_input), dim=1)

    return prediction_output


# def create_midi(prediction_output, output_file='output.mid'):
#     output_notes = []
#     total_offset = 0
#     for pattern, offset, duration in prediction_output:
#         if ('.' in pattern) or pattern.isdigit():
#             notes_in_chord = pattern.split('.')
#             notes = []
#             for current_note in notes_in_chord:
#                 new_note = note.Note(int(current_note))
#                 new_note.storedInstrument = instrument.Piano()
#                 notes.append(new_note)
#             new_chord = chord.Chord(notes)
#             new_chord.offset = total_offset
#             new_chord.duration = duration.Duration(float(duration))
#             output_notes.append(new_chord)
#         else:
#             new_note = note.Note(pattern)
#             new_note.offset = total_offset
#             new_note.duration = duration.Duration(float(duration))
#             new_note.storedInstrument = instrument.Piano()
#             output_notes.append(new_note)
#
#         total_offset += offset
#
#     midi_stream = stream.Stream(output_notes)
#     midi_stream.write('midi', fp=output_file)


def create_midi(prediction_output, output_file='output.mid'):
    output_notes = []
    total_offset = 0

    inst = instrument.Instrument()
    inst.midiProgram = 81
    for pattern, offset, duration_value in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = inst
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = total_offset
            new_chord.duration = duration.Duration(duration_value)
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = total_offset
            new_note.duration = duration.Duration(duration_value)
            new_note.storedInstrument = inst
            output_notes.append(new_note)

        total_offset += offset

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load = False

    if load is False:
        note_data = get_notes('data/')
    else:
        with open('models/training_data.pkl', 'rb') as f:
            note_data, network_data = pickle.load(f)

    network_data = prepare_sequences(note_data)

    if load is False:
        with open('models/training_data.pkl', 'wb') as f:
            pickle.dump((note_data, network_data), f)

    # Create Dataset
    dataset = MusicDataset(network_data)

    # Split dataset into training and validation set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Initialize the model
    model = EmbConv2LSTM(note_data)
    model = model.to(device)

    # Define the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, amsgrad=True)

    # Define parameters for saving and generating MIDI
    save_interval = 50  # Save every 50 epochs
    generate_interval = 10  # Generate MIDI every 10 epochs

    # Training loop
    epochs = 10000
    for epoch in range(epochs):
        t_loss = train(model, train_dataloader, criterion, optimizer, device, note_data)
        v_loss = validate(model, val_dataloader, criterion, device, note_data)
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f"Train Loss:{t_loss:.4f}\t|\tValid Loss:{v_loss:.4f}")

        # Save model and optimizer every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"models/checkpoint_epoch_{epoch + 1}.pth")

        # Generate MIDI every generate_interval epochs
        if (epoch + 1) % generate_interval == 0:
            generate_midi(model, note_data, network_data, f'midi/training_gen{epoch + 1}.midi')


if __name__ == "__main__":
    main()
