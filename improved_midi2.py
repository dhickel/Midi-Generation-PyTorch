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


class MusicDataset(Dataset):
    def __init__(self, network_input, network_output, network_output_durations):
        self.network_input = network_input
        self.network_output = network_output
        self.network_output_durations = network_output_durations

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], (self.network_output[idx], self.network_output_durations[idx])



def contains(lst, item):
    for i in range(0, len(lst)):
        if lst[i] == item:
            return i

    return None

def get_notes():
    training_notes = []
    note_lookups = ["P"]

    for file in glob.glob("data/*.mid"):
        print("Parsing: ", file)
        midi = None
        notes_to_parse = None

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
            gap_duration = event.offset - prev_event_end  # Calc gap
            if gap_duration > 0:
                training_notes.append((0, gap_duration))  # add gap token, 0 is the token as added at the top of the function

            if isinstance(event, note.Note) or isinstance(event, chord.Chord):
                inst = event.activeSite.getInstrument()
                if inst and inst.midiProgram is not None and inst.midiProgram >= 110:  # Filter out percussion
                    continue

                if isinstance(event, note.Note):
                    n = str(event.pitch.nameWithOctave)
                    d = float(event.duration.quarterLength)
                    note_val = contains(note_lookups, n)

                    if note_val is None:
                        note_lookups.append(n)
                        note_val = len(note_lookups) - 1

                    training_notes.append((note_val, d))


                elif isinstance(event, chord.Chord):
                    c = '.'.join(str(n.nameWithOctave) for n in event.pitches)
                    d = float(event.duration.quarterLength)
                    chord_val = contains(note_lookups, c)

                    if chord_val is None:
                        note_lookups.append(c)
                        chord_val = len(note_lookups) - 1
                    training_notes.append((chord_val, d))

            prev_event_end = event.offset + event.duration.quarterLength

    return training_notes, note_lookups


def prepare_sequences(training_notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 32

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(training_notes) - sequence_length, 1):
        sequence_in = training_notes[i:i + sequence_length]
        sequence_out = training_notes[i + sequence_length]

        network_input.append(sequence_in)
        network_output.append(sequence_out)

    n_patterns = len(network_input)

    # Convert the lists to a NumPy array and reshape them
    network_input_array = np.array(network_input).reshape(n_patterns, sequence_length, 2)  # 2 for note and duration
    network_output_array = np.array(network_output).reshape(n_patterns, 2)

    # Convert the NumPy arrays to PyTorch tensors
    network_input = torch.tensor(network_input_array, dtype=torch.float32)
    network_output = torch.tensor(network_output_array, dtype=torch.float32)

    # Normalize input
    network_input /= float(n_vocab - 1)

    return network_input, network_output, n_vocab

def generate_notes(model, network_input, note_lookups, device):
    """ Generate notes from the neural network based on a sequence of notes """
    start = np.random.randint(0, len(network_input) - 1)

    pattern = network_input[start].unsqueeze(0).to(device)

    prediction_output = []

    for note_index in range(200):
        prediction, prediction_duration = model(pattern)

        _, index = torch.max(prediction, dim=1)
        _, duration_index = torch.max(prediction_duration, dim=1)

        result = (note_lookups[index.item()], duration_index.item())
        prediction_output.append(result)

        next_input = torch.tensor([[[index.item(), duration_index.item()]]], dtype=torch.float32).to(device)

        pattern = torch.cat((pattern[:, 1:, :], next_input), dim=1)

    return prediction_output


def create_midi(prediction_output, output_file='output.mid'):
    offset = 0
    output_notes = []
    for pattern, duration in prediction_output:
        if pattern == "P":
            offset += duration
            continue

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

        offset += duration

    print(output_notes)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)


def generate_midi(model, note_lookups, network_input, output_file='output.mid'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    prediction_output = generate_notes(model, network_input, note_lookups, device)
    create_midi(prediction_output, output_file)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_notes = 0
    correct_durations = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets_notes, targets_durations = targets[0].to(device), targets[1].to(device)

        optimizer.zero_grad()

        outputs_notes, outputs_durations = model(inputs)
        loss1 = criterion(outputs_notes, targets_notes.long())  # targets need to be long type for CrossEntropyLoss
        loss2 = criterion(outputs_durations, targets_durations.long())
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted_notes = torch.max(outputs_notes.data, 1)
        _, predicted_durations = torch.max(outputs_durations.data, 1)
        total += targets_notes.size(0)
        correct_notes += (predicted_notes == targets_notes).sum().item()
        correct_durations += (predicted_durations == targets_durations).sum().item()

    accuracy_notes = correct_notes / total
    accuracy_durations = correct_durations / total
    return running_loss / len(dataloader), accuracy_notes, accuracy_durations


def validate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()  # Define your criterion here
    running_loss = 0.0
    correct_notes = 0
    correct_durations = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets_notes, targets_durations = targets[0].to(device), targets[1].to(device)

            outputs_notes, outputs_durations = model(inputs)
            loss1 = criterion(outputs_notes, targets_notes.long())  # targets need to be long type for CrossEntropyLoss
            loss2 = criterion(outputs_durations, targets_durations.long())
            loss = loss1 + loss2

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted_notes = torch.max(outputs_notes.data, 1)
            _, predicted_durations = torch.max(outputs_durations.data, 1)
            total += targets_notes.size(0)
            correct_notes += (predicted_notes == targets_notes).sum().item()
            correct_durations += (predicted_durations == targets_durations).sum().item()

        accuracy_notes = correct_notes / total
        accuracy_durations = correct_durations / total
        return running_loss / len(dataloader), accuracy_notes, accuracy_durations


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_vocab, device):
        super(CNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.relu = nn.Mish()

        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=0.15)
        self.fc = nn.Linear(hidden_size, n_vocab)
        self.fc_duration = nn.Linear(hidden_size, n_vocab)  # It would be best if you had a separate n_vocab for durations
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out_notes = self.fc(out)
        out_durations = self.fc_duration(out)

        return out_notes, out_durations




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Get notes
    training_notes, note_lookups = get_notes()

    print(training_notes)
    n_vocab = len(note_lookups) + 1
    # Prepare sequences
    network_input, network_output, n_vocab = prepare_sequences(training_notes, n_vocab)

    # Create Dataset
    dataset = MusicDataset(network_input, network_output[:, 0], network_output[:, 1])

    # Split dataset into training and validation set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    with open('training_data.pkl', 'wb') as f:
        pickle.dump((train_dataset, val_dataset), f)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Initialize the model
    model = CNNLSTMModel(input_size=2, hidden_size=256, num_layers=2, n_vocab=n_vocab, device=device)
    model = model.to(device)

    # Define the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Define parameters for saving and generating MIDI
    save_interval = 50  # Save every 50 epochs
    generate_interval = 50  # Generate MIDI every 10 epochs

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        train_loss, train_accuracy_notes, train_accuracy_durations = train(model, train_dataloader, criterion,
                                                                            optimizer, device)
        val_loss, val_accuracy_notes, val_accuracy_durations = validate(model, val_dataloader, device)
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss     : {train_loss:.3f}\t|\tValid Loss     : {val_loss:.3f}')
        print(f'Train Acc Notes: {train_accuracy_notes:.3f}\t|\tValid Acc Notes : {val_accuracy_notes:.3f}')
        print(f'Valid Acc Durrs: {train_accuracy_durations:.3f}\t|\tValid Acc Durrs: {val_accuracy_durations:.3f}')

        # Save model and optimizer every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"checkpoint_epoch_{epoch + 1}.pth")

        # Generate MIDI every generate_interval epochs
        if (epoch + 1) % generate_interval == 0:
            generate_midi(model, note_lookups, network_input, f'midi/training_gen{epoch+1}.midi')




if __name__ == "__main__":
    main()

