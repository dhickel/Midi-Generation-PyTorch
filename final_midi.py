import mido
import numpy as np
import glob
import torch
from mido import Message, MidiFile, MidiTrack
from music21 import converter, instrument, note, chord, stream, duration, pitch
import pickle
from torch import optim, nn
from torch.nn import Conv1d, MSELoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from Data import MusicDataset, NoteData, NetworkData



class MusicDataset(Dataset):
    def __init__(self, network_data):
        self.network_input = network_data.input
        self.network_output_notes = network_data.output_notes
        self.network_output_offsets = network_data.output_offsets
        self.network_output_durations = network_data.output_durations

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], (self.network_output_notes[idx], self.network_output_offsets[idx], self.network_output_durations[idx])


class NetworkData():
    def __init__(self, network_input, network_output_notes,network_output_offsets,network_output_durations):
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

        prev_event_end = 0
        for event in notes_to_parse:

            if isinstance(event, note.Note) or isinstance(event, chord.Chord):
                offset = event.offset - prev_event_end  # Calc gap
                if offset < 0:
                    offset = 0
                inst = event.activeSite.getInstrument()
                if inst and inst.midiProgram is not None and inst.midiProgram >= 110:  # Filter out percussion
                    continue
                note_val = None
                durr_val = None
                if isinstance(event, note.Note):
                    note_val = str(event.pitch.nameWithOctave)
                    durr_val = float(event.duration.quarterLength)
                elif isinstance(event, chord.Chord):
                    note_val = '.'.join(str(p.nameWithOctave) for p in event.pitches)
                    durr_val = float(event.duration.quarterLength)

                if note_val is not None and durr_val is not None:
                    note_val = data.add_note_if_absent(note_val)
                    durr_val = data.add_durr_if_absent(durr_val)
                    offset = data.add_offs_if_absent(offset)
                    data.training_notes.append((note_val, offset, durr_val))
                prev_event_end = event.offset + event.duration.quarterLength

        data.calc_max()
        print(f'Offsets{data.offset_table}')
    return data


#


def prepare_sequences(note_data):
    sequence_length = 32

    network_input = []
    network_output_notes = []
    network_output_offsets = []
    network_output_durations = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(note_data.training_notes) - sequence_length, 1):
        sequence_in = note_data.training_notes[i:i + sequence_length]
        sequence_out = note_data.training_notes[i + sequence_length]
        network_input.append([[x[0], x[1], x[2]] for x in sequence_in])  # Include all channels in input
        network_output_notes.append(sequence_out[0])
        network_output_offsets.append(sequence_out[1])
        network_output_durations.append(sequence_out[2])

    n_patterns = len(network_input)

    # Convert lists to NumPy arrays
    network_input_array = np.array(network_input)

    network_output_notes_array = np.array(network_output_notes)
    network_output_offsets_array = np.array(network_output_offsets)
    network_output_durations_array = np.array(network_output_durations)

    # Convert the NumPy arrays to PyTorch tensors
    network_input = torch.tensor(network_input_array, dtype=torch.float32)
    network_output_notes = torch.tensor(network_output_notes_array, dtype=torch.float32)
    network_output_offsets = torch.tensor(network_output_offsets_array, dtype=torch.float32)
    network_output_durations = torch.tensor(network_output_durations_array, dtype=torch.float32)

    # Normalize input and output
    network_input /= torch.tensor([float(note_data.n_vocab() - 1), float(note_data.o_vocab() - 1), float(note_data.d_vocab() - 1)], dtype=torch.float32)
    network_output_notes /= float(note_data.n_vocab() - 1)
    network_output_offsets /= float(note_data.o_vocab() - 1)
    network_output_durations /= float(note_data.d_vocab() - 1)

    return NetworkData(network_input, network_output_notes, network_output_offsets, network_output_durations)



def generate_midi(model, note_data, network_data, output_file='output.mid'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    prediction_output = generate_notes(model, note_data, network_data, device)
    create_midi(prediction_output, output_file)


def generate_notes(model, note_data, network_data, device):
    """ Generate notes from the neural network based on a sequence of notes """
    model.eval()
    start = np.random.randint(0, len(network_data.input) - 1)

    pattern = network_data.input[start].unsqueeze(0).to(device)

    prediction_output = []
    print(prediction_output)

    for note_index in range(200):
        prediction_note, prediction_offset, prediction_duration = model(pattern)

        _, note_index = torch.max(prediction_note, dim=1)
        _, offset_index = torch.max(prediction_offset, dim=1)
        _, duration_index = torch.max(prediction_duration, dim=1)

        result = (note_data.note_table[note_index.item()], note_data.offset_table[offset_index.item()], note_data.duration_table[duration_index.item()])
        prediction_output.append(result)

        next_input = torch.tensor([[[note_index.item(), offset_index.item(), duration_index.item()]]], dtype=torch.float32).to(device)

        pattern = torch.cat((pattern[:, 1:, :], next_input), dim=1)

    return prediction_output


def create_midi(prediction_output, output_file='output.mid'):
    output_notes = []
    total_offset = 0
    for pattern, offset, duration in prediction_output:
        print(f'Total Offset:{total_offset}')
        print(f'Offset{offset}')
        print(f'pattern{pattern}')
        print(f'durration{duration}')
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        total_offset += offset

    print(output_notes)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_notes = 0
    correct_offsets = 0
    correct_durations = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets_notes, targets_offsets, targets_durations = targets[0].to(device), targets[1].to(device), targets[2].to(device)

        optimizer.zero_grad()

        outputs_notes, outputs_offsets, outputs_durations = model(inputs)
        loss1 = criterion(outputs_notes, targets_notes.long())  # targets need to be long type for CrossEntropyLoss
        loss2 = criterion(outputs_offsets, targets_offsets.long())
        loss3 = criterion(outputs_durations, targets_durations.long())
        loss = loss1 + loss2 + loss3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        running_loss += loss.item()

        _, predicted_notes = torch.max(outputs_notes.data, 1)
        _, predicted_offsets = torch.max(outputs_offsets.data, 1)
        _, predicted_durations = torch.max(outputs_durations.data, 1)
        # print(f"Predicted note:{predicted_notes}")
        # print(f"Predicted offset:{predicted_offsets}")
        # print(f"Predicted Duration:{predicted_durations}")

        total += targets_notes.size(0)
        correct_notes += (predicted_notes == targets_notes).sum().item()
        correct_offsets += (predicted_offsets == targets_offsets).sum().item()
        correct_durations += (predicted_durations == targets_durations).sum().item()
    print(f"Predicted note:{predicted_notes}")
    print(f"Predicted offset:{predicted_offsets}")
    print(f"Predicted Duration:{predicted_durations}")
    accuracy_notes = correct_notes / total
    accuracy_offsets = correct_offsets / total
    accuracy_durations = correct_durations / total
    return running_loss / len(dataloader), accuracy_notes, accuracy_offsets, accuracy_durations


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_notes = 0
    correct_offsets = 0
    correct_durations = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets_notes, targets_offsets, targets_durations = targets[0].to(device), targets[1].to(device), targets[2].to(device)

            outputs_notes, outputs_offsets, outputs_durations = model(inputs)
            loss1 = criterion(outputs_notes, targets_notes.long())  # targets need to be long type for CrossEntropyLoss
            loss2 = criterion(outputs_offsets, targets_offsets.long())
            loss3 = criterion(outputs_durations, targets_durations.long())
            loss = loss1 + loss2 + loss3

            running_loss += loss.item()

            _, predicted_notes = torch.max(outputs_notes.data, 1)
            _, predicted_offsets = torch.max(outputs_offsets.data, 1)
            _, predicted_durations = torch.max(outputs_durations.data, 1)
            total += targets_notes.size(0)
            correct_notes += (predicted_notes == targets_notes).sum().item()
            correct_offsets += (predicted_offsets == targets_offsets).sum().item()
            correct_durations += (predicted_durations == targets_durations).sum().item()

        accuracy_notes = correct_notes / total
        accuracy_offsets = correct_offsets / total
        accuracy_durations = correct_durations / total
        return running_loss / len(dataloader), accuracy_notes, accuracy_offsets, accuracy_durations


class CNNLSTMModel(nn.Module):
    # def __init__(self, note_data):
    #     super(CNNLSTMModel, self).__init__()
    #
    #     # define the properties
    #     self.note_vocab_size = note_data.n_vocab()
    #     self.offset_vocab_size = note_data.o_vocab()
    #     self.duration_vocab_size = note_data.d_vocab()
    #
    #     # define the layers
    #     self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
    #     self.lstm = nn.LSTM(64, 256, num_layers=2, batch_first=True)
    #     self.dropout = nn.Dropout(0.3)
    #     self.fc1 = nn.Linear(256, 128)
    #     self.fc2 = nn.Linear(128, note_data.n_vocab())
    #     self.fc3 = nn.Linear(128, note_data.o_vocab())
    #     self.fc4 = nn.Linear(128, note_data.d_vocab())

    def __init__(self, note_data):
        super(CNNLSTMModel, self).__init__()

        # define the properties
        self.note_vocab_size = note_data.n_vocab()
        self.offset_vocab_size = note_data.o_vocab()
        self.duration_vocab_size = note_data.d_vocab()

        # define the layers
        self.conv1 = nn.Conv1d(3,32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)  # BatchNorm layer after Conv2
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, note_data.n_vocab())
        self.fc3 = nn.Linear(64, note_data.o_vocab())
        self.fc4 = nn.Linear(64, note_data.d_vocab())

        # Initialize the weights of Conv and FC layers
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)
    def forward(self, x):
        # pass data through conv1
        #print("input tensor:", x.shape)
        #x = x.permute(0, 2, 1)
        x = x.transpose(1,2)
        x = F.relu(self.conv1(x))
        x = x.transpose(1, 2)

        # flatten the data for LSTM
        #x = x.view(x.size(0), -1, 64)
        #x = x.view(x.size(0), x.size(1), -1)

        # pass data through LSTM layers
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        # pass lstm output through fully connected layer
        x = self.fc1(lstm_out)
        # get the final output
        note_out = self.fc2(x)
        offset_out = self.fc3(x)
        duration_out = self.fc4(x)

        # return the final outputs
        return note_out, offset_out, duration_out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get notes
    # note_data = get_notes('data/split/')
    # print(note_data.training_notes)
    # # Prepare sequences
    # network_data = prepare_sequences(note_data)

    with open('training_data.pkl', 'rb') as f:
        note_data, network_data = pickle.load(f)

    # Create Dataset
    dataset = MusicDataset(network_data)

    # Split dataset into training and validation set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # with open('training_data.pkl', 'wb') as f:
    #     pickle.dump((note_data, network_data), f)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Initialize the model
    model = CNNLSTMModel(note_data)
    model = model.to(device)

    # Define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, amsgrad=True)

    # Define parameters for saving and generating MIDI
    save_interval = 50  # Save every 50 epochs
    generate_interval = 50  # Generate MIDI every 10 epochs

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        t_loss, t_acc_notes, t_acc_offsets, t_acc_durations = train(model, train_dataloader, criterion,
                                                                           optimizer, device)
        v_loss, v_acc_notes, v_acc_offsets, v_acc_durations = validate(model, val_dataloader, criterion, device)
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f"T Note Acc:{t_acc_notes:.4f}\t|\tV Note Acc:{v_acc_notes:.4f}")
        print(f"T Off Acc :{t_acc_offsets:.4f}\t|\tV Off Acc :{v_acc_offsets:.4f}")
        print(f"T Dur Acc :{t_acc_durations:.4f}\t|\tV Dur Acc :{v_acc_durations:.4f}")
        print(f"Train Loss:{t_loss:.4f}\t|\tValid Loss:{v_loss:.4f}")



        # Save model and optimizer every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"checkpoint_epoch_{epoch + 1}.pth")

        # Generate MIDI every generate_interval epochs
        if (epoch + 1) % generate_interval == 0:
            generate_midi(model,note_data, network_data, f'midi/training_gen{epoch + 1}.midi')


if __name__ == "__main__":
    main()
    # def __init__(self, note_data):
    #     super(ConvLSTM, self).__init__()
    #
    #     self.cnn = nn.Sequential(
    #         nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool1d(kernel_size=2, stride=2),
    #         nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool1d(kernel_size=2, stride=2),
    #         nn.Flatten()
    #     )
    #
    #     # dummy input to get the output size of CNN
    #     dummy_input = torch.ones(1, 3, 32)
    #     dummy_output = self.cnn(dummy_input)
    #     cnn_output_size = dummy_output.shape[-1]
    #
    #     self.lstm = nn.LSTM(cnn_output_size, 256, num_layers=2, batch_first=True, dropout=0.3)
    #     self.fc_note = nn.Linear(256, note_data.n_vocab)
    #     self.fc_offset = nn.Linear(256, note_data.o_vocab)
    #     self.fc_duration = nn.Linear(256, note_data.d_vocab)
    #
    #     self._initialize_weights()
    #
    #
    # def forward(self, x):
    #     batch_size, seq_length, _ = x.size()
    #
    #     # Treat each feature as a separate channel
    #     cnn_out = self.cnn(x.permute(0, 2, 1))  # (batch_size, num_channels, seq_length)
    #     cnn_out = cnn_out.view(batch_size, seq_length, -1)  # Reshape back for LSTM
    #
    #     lstm_out, _ = self.lstm(cnn_out)
    #     lstm_out = lstm_out[:, -1, :]  # Only take the output from the final timetep
    #     output_note = self.fc_note(lstm_out)
    #     output_offset = self.fc_offset(lstm_out)
    #     output_duration = self.fc_duration(lstm_out)
    #
    #     return output_note, output_offset, output_duration
