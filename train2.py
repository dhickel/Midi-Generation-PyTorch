import glob
import os

import numpy
import torch
from tqdm import tqdm

from music21 import converter, instrument, note, chord, stream, duration, pitch

import generation2
from data import NoteData, MidiDataset, NetworkData
from generation2 import create_midi_track
from model import *
import os
from torch.cuda.amp import autocast


def get_notes(directory, get_flat=False):
    data = NoteData()

    for file in glob.glob(f'{directory}*.mid'):
        print("Parsing: ", file)
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"Warning: could not parse {file}. Skipping. Error: {e}")
            continue

        if not get_flat:
            try:  # file has instrument parts
                instruments = instrument.partitionByInstrument(midi)
                notes_to_parse = instruments.parts[0].recurse()
            except:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
        else:
            notes_to_parse = midi.flat.notes

        prev_event_end = -(data.get_random_off())
        notes = []
        for event in notes_to_parse:

            if isinstance(event, note.Note) or isinstance(event, chord.Chord):
                offset = event.offset - prev_event_end  # Calc offset (distance from last note)
                if offset < 0:
                    offset = 0
                try:
                    inst = event.activeSite.getInstrument()
                    if inst and inst.midiProgram is not None and inst.midiProgram >= 110:  # Filter out percussion
                        continue
                except:
                    continue

                note_val = None
                durr_val = None
                vel_val = None

                if isinstance(event, note.Note):
                    note_val = str(event.pitch.nameWithOctave)
                    durr_val = float(event.duration.quarterLength)
                    vel_val = event.volume.velocity if event.volume else None

                elif isinstance(event, chord.Chord):
                    note_val = '.'.join(str(p.nameWithOctave) for p in event.pitches)
                    durr_val = float(event.duration.quarterLength)
                    vel_val = event.volume.velocity if event.volume else None

                if note_val is not None and durr_val is not None and vel_val is not None:
                    note_val = data.add_note_if_absent(note_val)
                    durr_val = data.add_durr_if_absent(durr_val)
                    offset = data.add_offs_if_absent(offset)
                    vel_val = data.add_vel_if_absent(vel_val)
                    notes.append((note_val, offset, durr_val, vel_val))

                    prev_event_end = event.offset

        data.training_notes.append(notes)

    data.calc_vocab()
    print(f'train:{data.training_notes}')
    print(f'Notes{data.note_table}')
    print(f'Offsets{data.offset_table}')
    print(f'Durations{data.duration_table}')
    print(f'Velocities{data.velocity_table}')
    return data


class Step:
    def __init__(self, max):
        self.channels = numpy.zeros((4, max), dtype=int)
        self.max = max
        self.idx = 0



    def add_note(self, note_tuple):
        if self.idx == self.max - 1:
            return

        for i in range(0, self.max):
            if self.channels[0][i] == note_tuple[0]:
                return

        for i in range (0,4):
            self.channels[i][self.idx] = note_tuple[i]

        self.idx += 1

    def get_step(self):
        # Bubble sort columns using first row as the key
        for i in range(self.max):
            for j in range(self.max - i - 1):
                if self.channels[0][j] > self.channels[0][j + 1]:
                    for k in range(4):
                        self.channels[k][j], self.channels[k][j + 1] = self.channels[k][j + 1], self.channels[k][j]

        # Rotate until zeros in first row are at the end
        while self.channels[0][0] == 0:
            self.channels = np.roll(self.channels, -1, axis=1)

        return self.channels



def get_notes_single(directory, max_chan, get_flat=True):
    data = NoteData()
    x = 0

    for file in glob.glob(f'{directory}*.mid'):
        print("Parsing: ", file)
        try:
            midi = converter.parse(file)
        except Exception as e:
            print(f"Warning: could not parse {file}. Skipping. Error: {e}")
            continue
        #
        # if not get_flat:
        #     try:  # file has instrument parts
        #         instruments = instrument.partitionByInstrument(midi)
        #         notes_to_parse = instruments.parts[0].recurse()
        #     except:  # file has notes in a flat structure
        #         notes_to_parse = midi.flat.notes
        # else:
        notes_to_parse = midi.flat.notes

        prev_event_offset = -(data.get_random_off())
        notes = []

        step = None
        for event in notes_to_parse:

            if isinstance(event, note.Note) or isinstance(event, chord.Chord):
                offset = event.offset - prev_event_offset  # Calc offset (distance from last note)

                if offset < 0:
                    offset = 0
                try:
                    inst = event.activeSite.getInstrument()
                    if inst and inst.midiProgram is not None and inst.midiProgram >= 110:  # Filter out percussion
                        continue
                except:
                    continue

                # if offset > 0:
                #     if step is not None:
                #         notes.append(step.get_step())
                #     step = Step(max_chan)

                if isinstance(event, note.Note):
                    # note_val = data.add_note_if_absent(str(event.pitch.nameWithOctave))
                    note_val = data.add_note_if_absent(event.pitch.midi)
                    durr_val = data.add_durr_if_absent(float(event.duration.quarterLength))
                    vel_val = data.add_vel_if_absent(event.volume.velocity) if event.volume else None
                    off_val = data.add_offs_if_absent(offset)
                    tup = (note_val, off_val, durr_val, vel_val)
                    if not any(item is None for item in tup):
                        notes.append(tup)


                elif isinstance(event, chord.Chord):
                    p = event.pitches

                    for i in range(0, len(event.pitches)):
                        note_val = data.add_note_if_absent(p[i].midi)
                        durr_val = data.add_durr_if_absent(event.duration.quarterLength)
                        vel_val = data.add_vel_if_absent(event.volume.velocity if event.volume else None)
                        off_val = data.add_offs_if_absent(offset)
                        tup = (note_val, off_val, durr_val, vel_val)
                        if not any(item is None for item in tup):
                            notes.append(tup)

                prev_event_offset = event.offset
        data.training_notes.append(notes)

        # pred =[]
        # for step in notes:
        #     ns = []
        #     os = []
        #     ds = []
        #     vs = []
        #     for i in range(6):
        #         ns.append(data.get_note(step[0][i]))
        #         os.append(data.get_offset(step[1][i]))
        #         ds.append(data.get_duration(step[2][i]))
        #         vs.append(data.get_velocity(step[3][i]))
        #     pred.append((ns,os,ds,vs))
        #
        # file = file.split("/")
        # output_file = f"test_{file[2]}.mid"
        # create_midi_track(pred, output_file=output_file)

        # x += 1
    data.calc_vocab()
    return data


# def prepare_sequences(note_data, device=torch.device("cpu"), sequence_length=64, skip_amount=1):
#     sequence_length = sequence_length
#
#     network_input = []
#     network_output_notes = []
#     network_output_offsets = []
#     network_output_durations = []
#     network_output_velocities = []
#
#     # create input sequences and the corresponding outputs
#     for notes in note_data.training_notes:
#         for i in range(0, len(notes) - sequence_length - 1, skip_amount):
#             sequence_in = notes[i:i + sequence_length]
#             sequence_out = notes[i + sequence_length]
#             network_input.append([[x[0], x[1], x[2], x[3]] for x in sequence_in])
#             network_output_notes.append(sequence_out[0])
#             network_output_offsets.append(sequence_out[1])
#             network_output_durations.append(sequence_out[2])
#             network_output_velocities.append(sequence_out[3])
#
#     network_input = torch.tensor(network_input, dtype=torch.long).to(device)
#     # Shape for cross_entropy: (N) where N is batch size.
#     # network_output_notes = torch.tensor(network_output_notes, torch.long).view(-1).to(device)
#     # network_output_offsets = torch.tensor(network_output_offsets, torch.long).view(-1).to(device)
#     # network_output_durations = torch.tensor(network_output_durations, torch.long).view(-1).to(device)
#     # network_output_velocities = torch.tensor(network_output_velocities, torch.long).view(-1).to(device)
#     network_output_notes = torch.tensor(network_output_notes, dtype=torch.long).to(device)
#     network_output_offsets = torch.tensor(network_output_offsets, dtype=torch.long).to(device)
#     network_output_durations = torch.tensor(network_output_durations, dtype=torch.long).to(device)
#     network_output_velocities = torch.tensor(network_output_velocities, dtype=torch.long).to(device)
#     return NetworkData(network_input, network_output_notes, network_output_offsets, network_output_durations,
#                        network_output_velocities)


def prepare_sequences(note_data, device=torch.device("cuda"), sequence_length=64, skip_amount=1):
    network_input = []
    network_output_notes = []
    network_output_offsets = []
    network_output_durations = []
    network_output_velocities = []

    # create input sequences and the corresponding outputs
    for notes in note_data.training_notes:
        for i in range(0, len(notes) - sequence_length - 1, skip_amount):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([[x[0], x[1], x[2], x[3]] for x in sequence_in])
            network_output_notes.append(sequence_out[0])
            network_output_offsets.append(sequence_out[1])
            network_output_durations.append(sequence_out[2])
            network_output_velocities.append(sequence_out[3])

    network_input = torch.tensor(network_input, dtype=torch.long).to(device)
    network_output_notes = torch.tensor(network_output_notes, dtype=torch.long).to(device)
    network_output_offsets = torch.tensor(network_output_offsets, dtype=torch.long).to(device)
    network_output_durations = torch.tensor(network_output_durations, dtype=torch.long).to(device)
    network_output_velocities = torch.tensor(network_output_velocities, dtype=torch.long).to(device)

    return NetworkData(network_input, network_output_notes, network_output_offsets, network_output_durations, network_output_velocities)




def generate_seed_from_int(seed_int, seq_length, note_data):
    # Create a random number generator with the provided seed
    rng = np.random.default_rng(seed_int)

    # Generate random indices within the range of each vocabulary
    note_indices = rng.integers(note_data.n_vocab, size=seq_length)
    offset_indices = rng.integers(note_data.o_vocab, size=seq_length)
    duration_indices = rng.integers(note_data.d_vocab, size=seq_length)
    velocity_indices = rng.integers(note_data.v_vocab, size=seq_length)

    # Stack the indices into a single sequence and reshape it to the required shape
    seed_sequence = np.vstack([note_indices, offset_indices, duration_indices, velocity_indices])
    seed_sequence = seed_sequence.T.reshape(1, seq_length, 4)

    # Convert to a PyTorch tensor
    seed_sequence = torch.tensor(seed_sequence, dtype=torch.float16)

    return seed_sequence


#
# def train(model, train_loader, criterion, optimizer, device, note_data, scheduler=None, clip_value=None):
#     model.train()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#     for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
#         inputs = inputs.to(device)
#         targets_note = targets_note.to(device)
#         targets_offset = targets_offset.to(device)
#         targets_duration = targets_duration.to(device)
#         targets_velocity = targets_velocity.to(device)
#
#         # Forward pass
#         output_note, output_offset, output_duration, output_velocity = model(inputs)
#
#         # Calculate loss
#         loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long())
#         loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long())
#         loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long())
#         loss_velocity = criterion(output_velocity.view(-1, note_data.v_vocab), targets_velocity.view(-1).long())
#
#         loss = loss_note + loss_offset + loss_duration + loss_velocity
#
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Gradient clip
#         if clip_value is not None:  # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#
#         optimizer.step()
#
#         if scheduler is not None:
#             scheduler.step()
#
#         running_loss += loss.item() * inputs.size(0)
#
#         # Calculate accuracy
#         _, predicted_notes = torch.max(output_note.data, 1)
#         _, predicted_offsets = torch.max(output_offset.data, 1)
#         _, predicted_durations = torch.max(output_duration.data, 1)
#         _, predicted_velocities = torch.max(output_velocity.data, 1)
#
#         total_predictions += targets_note.size(0)
#         correct_predictions += (predicted_notes == targets_note).sum().item()
#         correct_predictions += (predicted_offsets == targets_offset).sum().item()
#         correct_predictions += (predicted_durations == targets_duration).sum().item()
#         correct_predictions += (predicted_velocities == targets_velocity).sum().item()
#
#     accuracy = correct_predictions / (total_predictions * 4)
#
#     return (running_loss / len(train_loader.dataset)) / 4, accuracy

#
# def validate(model, valid_loader, criterion, device, note_data):
#     model.eval()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#     with torch.no_grad():
#         for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(valid_loader):
#             inputs = inputs.to(device)
#             targets_note = targets_note.to(device)
#             targets_offset = targets_offset.to(device)
#             targets_duration = targets_duration.to(device)
#             targets_velocity = targets_velocity.to(device)
#
#             # Forward pass
#             output_note, output_offset, output_duration, output_velocity = model(inputs)
#
#             # Calculate loss
#             loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long())
#             loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long())
#             loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long())
#             loss_velocity = criterion(output_velocity.view(-1, note_data.v_vocab), targets_velocity.view(-1).long())
#
#             loss = loss_note + loss_offset + loss_duration + loss_velocity
#
#             running_loss += loss.item() * inputs.size(0)
#
#             # Calculate accuracy
#             _, predicted_notes = torch.max(output_note.data, 1)
#             _, predicted_offsets = torch.max(output_offset.data, 1)
#             _, predicted_durations = torch.max(output_duration.data, 1)
#             _, predicted_velocities = torch.max(output_velocity.data, 1)
#
#             total_predictions += targets_note.size(0)
#             correct_predictions += (predicted_notes == targets_note).sum().item()
#             correct_predictions += (predicted_offsets == targets_offset).sum().item()
#             correct_predictions += (predicted_durations == targets_duration).sum().item()
#             correct_predictions += (predicted_velocities == targets_velocity).sum().item()
#
#         accuracy = correct_predictions / (total_predictions * 4)
#
#         return (running_loss / len(valid_loader.dataset)) / 4, accuracy

# def train(model, train_loader, criterion, optimizer, device, note_data, scaler, scheduler=None, clip_value=None):
#     model.train()
#     running_loss = 0.0
#     total_predictions = 0
#
#     for inputs, targets in tqdm(train_loader):
#         inputs = inputs.to(device)
#         targets = [target.to(device) for target in targets]  # move each target to the device
#
#         # Forward pass
#         with autocast():
#             outputs = model(inputs)  # outputs is a list of tensors
#
#             # Calculate loss
#
#             losses = [criterion(output, target) for output, target in zip(outputs, targets)]
#             loss = sum(losses)
#
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#
#             # Gradient clipping
#             if clip_value is not None:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#
#             scaler.step(optimizer)
#             scaler.update()
#
#             if scheduler is not None:
#                 scheduler.step()
#
#             running_loss += loss.item() * inputs.size(0)
#
#             # Calculate accuracy
#             correct_predictions = [(output.argmax(dim=1) == target).sum().item() for output, target in zip(outputs, targets)]
#             total_predictions += inputs.size(0)
#             accuracy = sum(correct_predictions) / (total_predictions * len(outputs))
#
#     return running_loss / len(train_loader.dataset), accuracy
#
#
#
# def validate(model, valid_loader, criterion, device, note_data):
#     model.eval()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#     with torch.no_grad():
#         for inputs, targets in tqdm(valid_loader):
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#
#             # Forward pass
#             with autocast():
#                 outputs = model(inputs)
#
#                 # Calculate loss
#                 loss = criterion(outputs, targets)
#
#             running_loss += loss.item() * inputs.size(0)
#
#             # Calculate accuracy
#             _, predicted = torch.max(outputs.data, 1)
#             total_predictions += targets.size(0)
#             correct_predictions += (predicted == targets).sum().item()
#
#         accuracy = correct_predictions / total_predictions
#
#         return running_loss / len(valid_loader.dataset), accuracy

#
# def train(model, train_loader, criterion, optimizer, device, note_data, scalar, scheduler=None, clip_value=None):
#     model.train()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#     for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
#         inputs = inputs.to(device)
#         targets_note = targets_note.to(device)
#         targets_offset = targets_offset.to(device)
#         targets_duration = targets_duration.to(device)
#         targets_velocity = targets_velocity.to(device)
#
#         # Forward pass
#         output_note, output_offset, output_duration, output_velocity = model(inputs.to(device))
#
#         # Calculate loss
#         loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long())
#         loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long())
#         loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long())
#         loss_velocity = criterion(output_velocity.view(-1, note_data.v_vocab), targets_velocity.view(-1).long())
#
#         loss = loss_note + loss_offset + loss_duration + loss_velocity
#
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Gradient clip
#         if clip_value is not None:  # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#
#         optimizer.step()
#
#         if scheduler is not None:
#             scheduler.step()
#
#         running_loss += loss.item() * inputs.size(0)
#
#         # Calculate accuracy
#         _, predicted_notes = torch.max(output_note.data, 1)
#         _, predicted_offsets = torch.max(output_offset.data, 1)
#         _, predicted_durations = torch.max(output_duration.data, 1)
#         _, predicted_velocities = torch.max(output_velocity.data, 1)
#
#         total_predictions += targets_note.size(0)
#         correct_predictions += (predicted_notes == targets_note).sum().item()
#         correct_predictions += (predicted_offsets == targets_offset).sum().item()
#         correct_predictions += (predicted_durations == targets_duration).sum().item()
#         correct_predictions += (predicted_velocities == targets_velocity).sum().item()
#
#     accuracy = correct_predictions / (total_predictions * 4)
#
#     return (running_loss / len(train_loader.dataset)) / 4, accuracy
# #
#
#
#
#
# Evaluation function
# def evaluate(model, val_loader, criterion, device, note_data):
#     model.eval()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#     with torch.no_grad():
#         for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(val_loader):
#             inputs = inputs.to(device)
#             targets_note = targets_note.to(device)
#             targets_offset = targets_offset.to(device)
#             targets_duration = targets_duration.to(device)
#             targets_velocity = targets_velocity.to(device)
#
#             # Forward pass
#             output_note, output_offset, output_duration, output_velocity = model(inputs)
#
#             # Calculate loss
#             loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long())
#             loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long())
#             loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long())
#             loss_velocity = criterion(output_velocity.view(-1, note_data.v_vocab), targets_velocity.view(-1).long())
#
#             loss = loss_note + loss_offset + loss_duration + loss_velocity
#
#             running_loss += loss.item() * inputs.size(0)
#
#             # Calculate accuracy
#             _, predicted_notes = torch.max(output_note.data, 1)
#             _, predicted_offsets = torch.max(output_offset.data, 1)
#             _, predicted_durations = torch.max(output_duration.data, 1)
#             _, predicted_velocities = torch.max(output_velocity.data, 1)
#
#             total_predictions += targets_note.size(0)
#             correct_predictions += (predicted_notes == targets_note).sum().item()
#             correct_predictions += (predicted_offsets == targets_offset).sum().item()
#             correct_predictions += (predicted_durations == targets_duration).sum().item()
#             correct_predictions += (predicted_velocities == targets_velocity).sum().item()
#
#     accuracy = correct_predictions / (total_predictions * 4)
#
#     return (running_loss / len(val_loader.dataset)) / 4, accuracy



# Training function
# def train(model, train_loader, criterion, optimizer, device, note_data, scalar,  bathc_size, scheduler=None, clip_value=2):
#     model.train()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#
#     hidden = model.init_hidden(device, batch_size=bathc_size)
#     for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
#         optimizer.zero_grad()
#       #  hidden = model.detach_hidden(hidden)
#
#
#
#         hidden = model.init_hidden(device, bathc_size)
#         inputs = inputs.to(device)
#         targets_note = targets_note.to(device)
#         targets_offset = targets_offset.to(device)
#         targets_duration = targets_duration.to(device)
#         targets_velocity = targets_velocity.to(device)
#
#
#         # Create weights for each target feature
#         weight_note = torch.where(targets_note != 0, 1.2, 0.2).to(device)
#         weight_offset = torch.where(targets_offset != 0, 1.2, 0.2).to(device)
#         weight_duration = torch.where(targets_duration != 0, 1, 0.2).to(device)
#         weight_velocity = torch.where(targets_velocity != 0, 0.5, 0.1).to(device)
#
#
#         # Forward pass
#         output_note, output_offset, output_duration, output_velocity, hidden = model(inputs.to(device),  hidden)
#
#         # Calculate loss for each prediction
#         loss_note = 0
#         loss_offset = 0
#         loss_duration = 0
#         loss_velocity = 0
#
#         for i in range(6):
#             element_loss_note = criterion(output_note[:, i, :], targets_note[:, i])
#             element_loss_offset = criterion(output_offset[:, i, :], targets_offset[:, i])
#             element_loss_duration = criterion(output_duration[:, i, :], targets_duration[:, i])
#             element_loss_velocity = criterion(output_velocity[:, i, :], targets_velocity[:, i])
#
#             weighted_loss_note = element_loss_note * weight_note[:, i]
#             weighted_loss_offset = element_loss_offset * weight_offset[:, i]
#             weighted_loss_duration = element_loss_duration * weight_duration[:, i]
#             weighted_loss_velocity = element_loss_velocity * weight_velocity[:, i]
#
#             loss_note += torch.mean(weighted_loss_note)
#             loss_offset += torch.mean(weighted_loss_offset)
#             loss_duration += torch.mean(weighted_loss_duration)
#             loss_velocity += torch.mean(weighted_loss_velocity)
#             #
#             # Calculate accuracy
#             _, predicted_notes = torch.max(output_note.data, 2)
#             _, predicted_offsets = torch.max(output_offset.data, 2)
#             _, predicted_durations = torch.max(output_duration.data, 2)
#             _, predicted_velocities = torch.max(output_velocity.data, 2)
#
#             total_predictions += targets_note.size(0) * targets_note.size(1)
#             correct_predictions += (predicted_notes == targets_note).sum().item()
#             correct_predictions += (predicted_offsets == targets_offset).sum().item()
#             correct_predictions += (predicted_durations == targets_duration).sum().item()
#             correct_predictions += (predicted_velocities == targets_velocity).sum().item()
#
#
#         loss = loss_note + loss_offset + loss_duration + loss_velocity
#
#
#         # Backward pass and optimization
#
#         loss.backward()
#
#         # Gradient clip
#         if clip_value is not None:  # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#
#         optimizer.step()
#
#         if scheduler is not None:
#             scheduler.step()
#
#         running_loss += loss.item() * inputs.size(0)
#
#
#
#     accuracy = correct_predictions / total_predictions
#
#     return running_loss / len(train_loader.dataset), accuracy
# def train(model, train_loader, criterion, optimizer, device, note_data, scalar, batch_size, scheduler=None, clip_value=2):
#     model.train()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#     hidden = model.init_hidden(device, batch_size=batch_size)
#     for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
#         optimizer.zero_grad()
#
#         hidden = model.init_hidden(device, batch_size)
#         inputs = inputs.to(device)
#         targets_note = targets_note.to(device)
#         targets_offset = targets_offset.to(device)
#         targets_duration = targets_duration.to(device)
#         targets_velocity = targets_velocity.to(device)
#
#         # Create weights for each target feature (once per batch)
#         weight_note = torch.where(targets_note != 0, 1, 0.2).to(device)
#         weight_offset = torch.where(targets_offset != 0, 1, 0.2).to(device)
#         weight_duration = torch.where(targets_duration != 0, 0.8, 0.2).to(device)
#         weight_velocity = torch.where(targets_velocity != 0, 0.5, 0.1).to(device)
#
#         # Forward pass
#         output_note, output_offset, output_duration, output_velocity, hidden = model(inputs.to(device), hidden)
#
#         # Calculate loss for each prediction
#         loss_note = 0
#         loss_offset = 0
#         loss_duration = 0
#         loss_velocity = 0
#
#         for i in range(6):
#             # Calculate loss for this timestamp
#             element_loss_note = criterion(output_note[:, i, :], targets_note[:, i])
#             element_loss_offset = criterion(output_offset[:, i, :], targets_offset[:, i])
#             element_loss_duration = criterion(output_duration[:, i, :], targets_duration[:, i])
#             element_loss_velocity = criterion(output_velocity[:, i, :], targets_velocity[:, i])
#
#             # Apply weights
#             weighted_loss_note = element_loss_note * weight_note[:, i]
#             weighted_loss_offset = element_loss_offset * weight_offset[:, i]
#             weighted_loss_duration = element_loss_duration * weight_duration[:, i]
#             weighted_loss_velocity = element_loss_velocity * weight_velocity[:, i]
#
#             # Accumulate losses
#             loss_note += torch.mean(weighted_loss_note)
#             loss_offset += torch.mean(weighted_loss_offset)
#             loss_duration += torch.mean(weighted_loss_duration)
#             loss_velocity += torch.mean(weighted_loss_velocity)
#
#         loss = loss_note + loss_offset + loss_duration + loss_velocity
#
#         loss.backward()
#
#         if clip_value is not None:  # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#
#         optimizer.step()
#
#         if scheduler is not None:
#             scheduler.step()
#
#         running_loss += loss.item() * inputs.size(0)
#
#         # Calculate accuracy
#         _, predicted_notes = torch.max(output_note.data, 2)
#         _, predicted_offsets = torch.max(output_offset.data, 2)
#         _, predicted_durations = torch.max(output_duration.data, 2)
#         _, predicted_velocities = torch.max(output_velocity.data, 2)
#
#         total_predictions += targets_note.size(0) * targets_note.size(1)
#         correct_predictions += (predicted_notes == targets_note).sum().item()
#         correct_predictions += (predicted_offsets == targets_offset).sum().item()
#         correct_predictions += (predicted_durations == targets_duration).sum().item()
#         correct_predictions += (predicted_velocities == targets_velocity).sum().item()
#
#     accuracy = correct_predictions / total_predictions
#
#     return running_loss / len(train_loader.dataset), accuracy

#
# def train(model, train_loader, criterion, optimizer, device, note_data, scalar,  batch_size, scheduler=None, clip_value=2):
#     model.train()
#     running_loss = 0.0
#     total_predictions = 0
#     correct_predictions = 0
#
#     hidden = model.init_hidden(device, batch_size=batch_size)
#     for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
#         optimizer.zero_grad()
#         hidden = model.detach_hidden(hidden)
#
#         inputs = inputs.to(device)
#         targets_note = targets_note.to(device)
#         targets_offset = targets_offset.to(device)
#         targets_duration = targets_duration.to(device)
#         targets_velocity = targets_velocity.to(device)
#
#         # Forward pass
#         output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)
#
#         # Create masks for each target feature
#         mask_note = (targets_note != 0).to(device)
#         mask_offset = (targets_offset != 0).to(device)
#         mask_duration = (targets_duration != 0).to(device)
#         mask_velocity = (targets_velocity != 0).to(device)
#
#         loss_note = 0
#         loss_offset = 0
#         loss_duration = 0
#         loss_velocity = 0
#         for i in range(6):
#             # Element-wise loss for each feature
#             element_loss_note = criterion(output_note[:, i, :], targets_note[:, i])
#             element_loss_offset = criterion(output_offset[:, i, :], targets_offset[:, i])
#             element_loss_duration = criterion(output_duration[:, i, :], targets_duration[:, i])
#             element_loss_velocity = criterion(output_velocity[:, i, :], targets_velocity[:, i])
#
#             # Apply the mask to the loss
#             masked_loss_note = element_loss_note * mask_note[:, i]
#             masked_loss_offset = element_loss_offset * mask_offset[:, i]
#             masked_loss_duration = element_loss_duration * mask_duration[:, i]
#             masked_loss_velocity = element_loss_velocity * mask_velocity[:, i]
#
#            # Mean of the masked losses
#             loss_note += torch.mean(masked_loss_note)
#             loss_offset += torch.mean(masked_loss_offset)
#             loss_duration += torch.mean(masked_loss_duration)
#             loss_velocity += torch.mean(masked_loss_velocity)
#             #
#             # loss_note += torch.sum(masked_loss_note)
#             # loss_offset += torch.sum(masked_loss_offset)
#             # loss_duration += torch.sum(masked_loss_duration)
#             # loss_velocity += torch.sum(masked_loss_velocity)
#
#         _, predicted_notes = torch.max(output_note.data, 2)
#         _, predicted_offsets = torch.max(output_offset.data, 2)
#         _, predicted_durations = torch.max(output_duration.data, 2)
#         _, predicted_velocities = torch.max(output_velocity.data, 2)
#         torch.sum
#         # Calculate accuracy only on non-padding tokens
#         correct_predictions += ((predicted_notes == targets_note) * mask_note).sum().item()
#         correct_predictions += ((predicted_offsets == targets_offset) * mask_offset).sum().item()
#         correct_predictions += ((predicted_durations == targets_duration) * mask_duration).sum().item()
#         correct_predictions += ((predicted_velocities == targets_velocity) * mask_velocity).sum().item()
#
#         total_predictions += mask_note.sum().item()
#         total_predictions += mask_offset.sum().item()
#         total_predictions += mask_duration.sum().item()
#         total_predictions += mask_velocity.sum().item()
#
#         loss = loss_note + loss_offset + loss_duration + loss_velocity
#
#
#         # Backward pass and optimize
#         loss.backward()
#         if clip_value is not None:  # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
#         optimizer.step()
#
#         if scheduler is not None:
#             scheduler.step()
#
#         running_loss += loss.item() * inputs.size(0)
#
#     accuracy = correct_predictions / total_predictions
#
#     return (running_loss / len(train_loader.dataset)) / 4, accuracy
#
#
#
def train(model, train_loader, criterion, optimizer, device, note_data, scalar,  batch_size, scheduler=None, clip_value=2):
    model.train()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0

    hidden = model.init_hidden(device, batch_size=batch_size)

    for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        inputs = inputs.to(device)
        targets_note = targets_note.to(device)
        targets_offset = targets_offset.to(device)
        targets_duration = targets_duration.to(device)
        targets_velocity = targets_velocity.to(device)

        # Forward pass
        output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)

        # Create weights for each target feature
        weight_note = torch.where(targets_note != 0, 1.0, 0.25).to(device)
        weight_offset = torch.where(targets_offset != 0, 1.0, 0.25).to(device)
        weight_duration = torch.where(targets_duration != 0, 1.0, 0.25).to(device)
        weight_velocity = torch.where(targets_velocity != 0, 1.0, 0.25).to(device)

        loss_note = 0
        loss_offset = 0
        loss_duration = 0
        loss_velocity = 0
        for i in range(6):
            # Element-wise loss for each feature
            element_loss_note = criterion(output_note[:, i, :], targets_note[:, i])
            element_loss_offset = criterion(output_offset[:, i, :], targets_offset[:, i])
            element_loss_duration = criterion(output_duration[:, i, :], targets_duration[:, i])
            element_loss_velocity = criterion(output_velocity[:, i, :], targets_velocity[:, i])

            # Apply the weights to the loss
            weighted_loss_note = element_loss_note * weight_note[:, i]
            weighted_loss_offset = element_loss_offset * weight_offset[:, i]
            weighted_loss_duration = element_loss_duration * weight_duration[:, i]
            weighted_loss_velocity = element_loss_velocity * weight_velocity[:, i]

            # Mean of the weighted losses
            loss_note += torch.mean(weighted_loss_note)
            loss_offset += torch.mean(weighted_loss_offset)
            loss_duration += torch.mean(weighted_loss_duration)
            loss_velocity += torch.mean(weighted_loss_velocity)

        _, predicted_notes = torch.max(output_note.data, 2)
        _, predicted_offsets = torch.max(output_offset.data, 2)
        _, predicted_durations = torch.max(output_duration.data, 2)
        _, predicted_velocities = torch.max(output_velocity.data, 2)

        # Calculate accuracy only on non-padding tokens
        correct_predictions += ((predicted_notes == targets_note) * weight_note).sum().item()
        correct_predictions += ((predicted_offsets == targets_offset) * weight_offset).sum().item()
        correct_predictions += ((predicted_durations == targets_duration) * weight_duration).sum().item()
        correct_predictions += ((predicted_velocities == targets_velocity) * weight_velocity).sum().item()

        total_predictions += weight_note.sum().item()
        total_predictions += weight_offset.sum().item()
        total_predictions += weight_duration.sum().item()
        total_predictions += weight_velocity.sum().item()

        loss = loss_note + loss_offset + loss_duration + loss_velocity

        # Backward pass and optimize
        loss.backward()
        if clip_value is not None:  # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)

    accuracy = correct_predictions / total_predictions

    return (running_loss / len(train_loader.dataset)) / 4, accuracy


# Evaluation function
def evaluate(model, val_loader, criterion, device, note_data, batch_size):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0



    hidden = model.init_hidden(device, batch_size=batch_size)
    with torch.no_grad():
        for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(val_loader):
            hidden = model.init_hidden(device, batch_size=batch_size)
            inputs = inputs.to(device)
            targets_note = targets_note.to(device)
            targets_offset = targets_offset.to(device)
            targets_duration = targets_duration.to(device)
            targets_velocity = targets_velocity.to(device)

            # Forward pass
            output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)

            # Calculate loss for each prediction
            loss_note = 0
            loss_offset = 0
            loss_duration = 0
            loss_velocity = 0
            for i in range(6):
                loss_note += criterion(output_note[:, i, :], targets_note[:, i])
                loss_offset += criterion(output_offset[:, i, :], targets_offset[:, i])
                loss_duration += criterion(output_duration[:, i, :], targets_duration[:, i])
                loss_velocity += criterion(output_velocity[:, i, :], targets_velocity[:, i])

            loss = loss_note + loss_offset + loss_duration + loss_velocity

            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted_notes = torch.max(output_note.data, 2)
            _, predicted_offsets = torch.max(output_offset.data, 2)
            _, predicted_durations = torch.max(output_duration.data, 2)
            _, predicted_velocities = torch.max(output_velocity.data, 2)

            total_predictions += targets_note.size(0) * targets_note.size(1)
            correct_predictions += (predicted_notes == targets_note).sum().item()
            correct_predictions += (predicted_offsets == targets_offset).sum().item()
            correct_predictions += (predicted_durations == targets_duration).sum().item()
            correct_predictions += (predicted_velocities == targets_velocity).sum().item()

    accuracy = correct_predictions / (total_predictions * 6)

    return (running_loss / len(val_loader.dataset)) / 4, accuracy

#
# def evaluate(model, val_loader, criterion, device, note_data, batch_size):
#     model.eval()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#     hidden = model.init_hidden(device, batch_size=batch_size)
#     with torch.no_grad():
#         for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(val_loader):
#             hidden = model.detach_hidden(hidden)
#             inputs = inputs.to(device)
#             targets_note = targets_note.to(device)
#             targets_offset = targets_offset.to(device)
#             targets_duration = targets_duration.to(device)
#             targets_velocity = targets_velocity.to(device)
#
#             # Forward pass
#             output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)
#
#             # Create masks for each target feature
#             mask_note = (targets_note != 0).to(device)
#             mask_offset = (targets_offset != 0).to(device)
#             mask_duration = (targets_duration != 0).to(device)
#             mask_velocity = (targets_velocity != 0).to(device)
#
#             loss_note = 0
#             loss_offset = 0
#             loss_duration = 0
#             loss_velocity = 0
#             for i in range(6):
#                 # Element-wise loss for each feature
#                 element_loss_note = criterion(output_note[:, i, :], targets_note[:, i])
#                 element_loss_offset = criterion(output_offset[:, i, :], targets_offset[:, i])
#                 element_loss_duration = criterion(output_duration[:, i, :], targets_duration[:, i])
#                 element_loss_velocity = criterion(output_velocity[:, i, :], targets_velocity[:, i])
#
#                 # Apply the mask to the loss
#                 masked_loss_note = element_loss_note * mask_note[:, i]
#                 masked_loss_offset = element_loss_offset * mask_offset[:, i]
#                 masked_loss_duration = element_loss_duration * mask_duration[:, i]
#                 masked_loss_velocity = element_loss_velocity * mask_velocity[:, i]
#
#                 # Mean of the masked losses
#                 loss_note += torch.mean(masked_loss_note)
#                 loss_offset += torch.mean(masked_loss_offset)
#                 loss_duration += torch.mean(masked_loss_duration)
#                 loss_velocity += torch.mean(masked_loss_velocity)
#
#             loss = loss_note + loss_offset + loss_duration + loss_velocity
#
#             _, predicted_notes = torch.max(output_note.data, 2)
#             _, predicted_offsets = torch.max(output_offset.data, 2)
#             _, predicted_durations = torch.max(output_duration.data, 2)
#             _, predicted_velocities = torch.max(output_velocity.data, 2)
#
#             # Calculate accuracy only on non-padding tokens
#             correct_predictions += ((predicted_notes == targets_note) * mask_note).sum().item()
#             correct_predictions += ((predicted_offsets == targets_offset) * mask_offset).sum().item()
#             correct_predictions += ((predicted_durations == targets_duration) * mask_duration).sum().item()
#             correct_predictions += ((predicted_velocities == targets_velocity) * mask_velocity).sum().item()
#
#             total_predictions += mask_note.sum().item()
#             total_predictions += mask_offset.sum().item()
#             total_predictions += mask_duration.sum().item()
#             total_predictions += mask_velocity.sum().item()
#
#             running_loss += loss.item() * inputs.size(0)
#
#     accuracy = correct_predictions / total_predictions
#
#     return running_loss / len(val_loader.dataset), accuracy
#
def evaluate(model, dev_loader, criterion, device, note_data, batch_size):
    model.eval()
    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0

    hidden = model.init_hidden(device, batch_size=batch_size)

    with torch.no_grad():
        for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(dev_loader):
            inputs = inputs.to(device)
            targets_note = targets_note.to(device)
            targets_offset = targets_offset.to(device)
            targets_duration = targets_duration.to(device)
            targets_velocity = targets_velocity.to(device)

            # Forward pass
            output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)

            # Create weights for each target feature
            weight_note = torch.where(targets_note != 0, 1.0, 0.25).to(device)
            weight_offset = torch.where(targets_offset != 0, 1, 0.25).to(device)
            weight_duration = torch.where(targets_duration != 0, 1, 0.25).to(device)
            weight_velocity = torch.where(targets_velocity != 0, 1, 0.25).to(device)

            loss_note = 0
            loss_offset = 0
            loss_duration = 0
            loss_velocity = 0
            for i in range(6):
                # Element-wise loss for each feature
                element_loss_note = criterion(output_note[:, i, :], targets_note[:, i])
                element_loss_offset = criterion(output_offset[:, i, :], targets_offset[:, i])
                element_loss_duration = criterion(output_duration[:, i, :], targets_duration[:, i])
                element_loss_velocity = criterion(output_velocity[:, i, :], targets_velocity[:, i])

                # Apply the weights to the loss
                weighted_loss_note = element_loss_note * weight_note[:, i]
                weighted_loss_offset = element_loss_offset * weight_offset[:, i]
                weighted_loss_duration = element_loss_duration * weight_duration[:, i]
                weighted_loss_velocity = element_loss_velocity * weight_velocity[:, i]

                # Mean of the weighted losses
                loss_note += torch.mean(weighted_loss_note)
                loss_offset += torch.mean(weighted_loss_offset)
                loss_duration += torch.mean(weighted_loss_duration)
                loss_velocity += torch.mean(weighted_loss_velocity)

            _, predicted_notes = torch.max(output_note.data, 2)
            _, predicted_offsets = torch.max(output_offset.data, 2)
            _, predicted_durations = torch.max(output_duration.data, 2)
            _, predicted_velocities = torch.max(output_velocity.data, 2)

            # Calculate accuracy only on non-padding tokens
            correct_predictions += ((predicted_notes == targets_note) * weight_note).sum().item()
            correct_predictions += ((predicted_offsets == targets_offset) * weight_offset).sum().item()
            correct_predictions += ((predicted_durations == targets_duration) * weight_duration).sum().item()
            correct_predictions += ((predicted_velocities == targets_velocity) * weight_velocity).sum().item()

            total_predictions += weight_note.sum().item()
            total_predictions += weight_offset.sum().item()
            total_predictions += weight_duration.sum().item()
            total_predictions += weight_velocity.sum().item()

            loss = loss_note + loss_offset + loss_duration + loss_velocity

            running_loss += loss.item() * inputs.size(0)

    accuracy = correct_predictions / total_predictions

    return (running_loss / len(dev_loader.dataset)) / 4, accuracy
# def evaluate(model, val_loader, criterion, device, note_data, batch_size):
#     model.eval()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#     hidden = model.init_hidden(device, batch_size=batch_size)
#     with torch.no_grad():
#         for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(val_loader):
#             hidden = model.init_hidden(device, batch_size=batch_size)
#             inputs = inputs.to(device)
#             targets_note = targets_note.to(device)
#             targets_offset = targets_offset.to(device)
#             targets_duration = targets_duration.to(device)
#             targets_velocity = targets_velocity.to(device)
#
#             # Forward pass
#             output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)
#
#             # Create weights for each target feature (once per batch)
#             weight_note = torch.where(targets_note != 0, 1.0, 0.25).to(device)
#             weight_offset = torch.where(targets_offset != 0, 1.0, 0.25).to(device)
#             weight_duration = torch.where(targets_duration != 0, 1.0, 0.25).to(device)
#             weight_velocity = torch.where(targets_velocity != 0, 1.0, 0.25).to(device)
#
#             loss_note = 0
#             loss_offset = 0
#             loss_duration = 0
#             loss_velocity = 0
#             for i in range(6):
#                 # Element-wise loss for each feature
#                 element_loss_note = criterion(output_note[:, i, :], targets_note[:, i])
#                 element_loss_offset = criterion(output_offset[:, i, :], targets_offset[:, i])
#                 element_loss_duration = criterion(output_duration[:, i, :], targets_duration[:, i])
#                 element_loss_velocity = criterion(output_velocity[:, i, :], targets_velocity[:, i])
#
#                 # Apply the weights to the loss
#                 weighted_loss_note = element_loss_note * weight_note[:, i]
#                 weighted_loss_offset = element_loss_offset * weight_offset[:, i]
#                 weighted_loss_duration = element_loss_duration * weight_duration[:, i]
#                 weighted_loss_velocity = element_loss_velocity * weight_velocity[:, i]
#
#                 # Mean of the weighted losses
#                 loss_note += torch.mean(weighted_loss_note)
#                 loss_offset += torch.mean(weighted_loss_offset)
#                 loss_duration += torch.mean(weighted_loss_duration)
#                 loss_velocity += torch.mean(weighted_loss_velocity)
#
#             loss = loss_note + loss_offset + loss_duration + loss_velocity
#
#             _, predicted_notes = torch.max(output_note.data, 2)
#             _, predicted_offsets = torch.max(output_offset.data, 2)
#             _, predicted_durations = torch.max(output_duration.data, 2)
#             _, predicted_velocities = torch.max(output_velocity.data, 2)
#
#             # Calculate accuracy only on non-padding tokens
#             total_predictions += targets_note.size(0) * targets_note.size(1)
#             correct_predictions += (predicted_notes == targets_note).sum().item()
#             correct_predictions += (predicted_offsets == targets_offset).sum().item()
#             correct_predictions += (predicted_durations == targets_duration).sum().item()
#             correct_predictions += (predicted_velocities == targets_velocity).sum().item()
#
#             running_loss += loss.item() * inputs.size(0)
#
#     accuracy = correct_predictions / total_predictions
#
#     return running_loss / len(val_loader.dataset), accuracy

def train(model, train_loader, criterion, optimizer, device, note_data, scalar, batch_size,scheduler=None, clip_value=None):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    hidden = model.init_hidden(device, batch_size)
    for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
        inputs = inputs.to(device)
        targets_note = targets_note.to(device)
        targets_offset = targets_offset.to(device)
        targets_duration = targets_duration.to(device)
        targets_velocity = targets_velocity.to(device)

        # Forward pass
        output_note, output_offset, output_duration, output_velocity, _  = model(inputs.to(device, hidden))

        # Calculate loss
        loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long())
        loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long())
        loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long())
        loss_velocity = criterion(output_velocity.view(-1, note_data.v_vocab), targets_velocity.view(-1).long())

        loss = loss_note + loss_offset + loss_duration + loss_velocity

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Gradient clip
        if clip_value is not None:  # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy
        _, predicted_notes = torch.max(output_note.data, 1)
        _, predicted_offsets = torch.max(output_offset.data, 1)
        _, predicted_durations = torch.max(output_duration.data, 1)
        _, predicted_velocities = torch.max(output_velocity.data, 1)

        total_predictions += targets_note.size(0)
        correct_predictions += (predicted_notes == targets_note).sum().item()
        correct_predictions += (predicted_offsets == targets_offset).sum().item()
        correct_predictions += (predicted_durations == targets_duration).sum().item()
        correct_predictions += (predicted_velocities == targets_velocity).sum().item()

    accuracy = correct_predictions / (total_predictions * 4)

    return (running_loss / len(train_loader.dataset)) / 4, accuracy



#Evaluation function
def train(model, train_loader, criterion, optimizer, device, note_data, scalar, batch_size, scheduler=None, clip_value=None):
    model.train()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0

   # hidden = model.init_hidden(device, batch_size=batch_size)

    for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
        optimizer.zero_grad()
        hidden = model.init_hidden(device, batch_size)

        inputs = inputs.to(device)
        targets_note = targets_note.to(device)
        targets_offset = targets_offset.to(device)
        targets_duration = targets_duration.to(device)
        targets_velocity = targets_velocity.to(device)

        # Forward pass
        output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)

        loss_note = 0
        loss_offset = 0
        loss_duration = 0
        loss_velocity = 0
        for i in range(6):
            # Element-wise loss for each feature
            loss_note += criterion(output_note[:, i, :], targets_note[:, i])
            loss_offset += criterion(output_offset[:, i, :], targets_offset[:, i])
            loss_duration += criterion(output_duration[:, i, :], targets_duration[:, i])
            loss_velocity += criterion(output_velocity[:, i, :], targets_velocity[:, i])

        _, predicted_notes = torch.max(output_note.data, 2)
        _, predicted_offsets = torch.max(output_offset.data, 2)
        _, predicted_durations = torch.max(output_duration.data, 2)
        _, predicted_velocities = torch.max(output_velocity.data, 2)

        # Calculate accuracy
        correct_predictions += (predicted_notes == targets_note).sum().item()
        correct_predictions += (predicted_offsets == targets_offset).sum().item()
        correct_predictions += (predicted_durations == targets_duration).sum().item()
        correct_predictions += (predicted_velocities == targets_velocity).sum().item()

        total_predictions += targets_note.numel()
        total_predictions += targets_offset.numel()
        total_predictions += targets_duration.numel()
        total_predictions += targets_velocity.numel()

        loss = loss_note + loss_offset + loss_duration + loss_velocity

        # Backward pass and optimize
        loss.backward()
        if clip_value is not None:  # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)

    accuracy = correct_predictions / total_predictions

    return (running_loss / len(train_loader.dataset)) / 4, accuracy


def evaluate(model, val_loader, criterion, device, note_data, batch_size):
    model.eval()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0

    hidden = model.init_hidden(device, batch_size=batch_size)

    with torch.no_grad():
        for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(val_loader):
            hidden = model.detach_hidden(hidden)

            inputs = inputs.to(device)
            targets_note = targets_note.to(device)
            targets_offset = targets_offset.to(device)
            targets_duration = targets_duration.to(device)
            targets_velocity = targets_velocity.to(device)

            # Forward pass
            output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)

            loss_note = 0
            loss_offset = 0
            loss_duration = 0
            loss_velocity = 0
            for i in range(6):
                # Element-wise loss for each feature
                loss_note += criterion(output_note[:, i, :], targets_note[:, i]).mean()
                loss_offset += criterion(output_offset[:, i, :], targets_offset[:, i]).mean()
                loss_duration += criterion(output_duration[:, i, :], targets_duration[:, i]).mean()
                loss_velocity += criterion(output_velocity[:, i, :], targets_velocity[:, i]).mean()

            _, predicted_notes = torch.max(output_note.data, 2)
            _, predicted_offsets = torch.max(output_offset.data, 2)
            _, predicted_durations = torch.max(output_duration.data, 2)
            _, predicted_velocities = torch.max(output_velocity.data, 2)

            # Calculate accuracy
            correct_predictions += (predicted_notes == targets_note).sum().item()
            correct_predictions += (predicted_offsets == targets_offset).sum().item()
            correct_predictions += (predicted_durations == targets_duration).sum().item()
            correct_predictions += (predicted_velocities == targets_velocity).sum().item()

            total_predictions += targets_note.numel()
            total_predictions += targets_offset.numel()
            total_predictions += targets_duration.numel()
            total_predictions += targets_velocity.numel()

            loss = loss_note + loss_offset + loss_duration + loss_velocity

            running_loss += loss.item() * inputs.size(0)

    accuracy = correct_predictions / total_predictions

    return (running_loss / len(val_loader.dataset)) / 4, accuracy



def train(model, train_loader, criterion, optimizer, device, batch_size, scheduler=None, clip_value=None):
    model.train()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0

    for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
        optimizer.zero_grad()
        hidden = model.init_hidden(device, batch_size)

        inputs = inputs.to(device)
        targets_note = targets_note.to(device)
        targets_offset = targets_offset.to(device)
        targets_duration = targets_duration.to(device)
        targets_velocity = targets_velocity.to(device)

        # Forward pass
        output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)

        # Element-wise loss for each feature
        loss_note = criterion(output_note, targets_note)
        loss_offset = criterion(output_offset, targets_offset)
        loss_duration = criterion(output_duration, targets_duration)
        loss_velocity = criterion(output_velocity, targets_velocity)

        # Calculate accuracy
        _, predicted_notes = torch.max(output_note.data, 1)
        _, predicted_offsets = torch.max(output_offset.data, 1)
        _, predicted_durations = torch.max(output_duration.data, 1)
        _, predicted_velocities = torch.max(output_velocity.data, 1)

        correct_predictions += (predicted_notes == targets_note).sum().item()
        correct_predictions += (predicted_offsets == targets_offset).sum().item()
        correct_predictions += (predicted_durations == targets_duration).sum().item()
        correct_predictions += (predicted_velocities == targets_velocity).sum().item()

        total_predictions += targets_note.numel()
        total_predictions += targets_offset.numel()
        total_predictions += targets_duration.numel()
        total_predictions += targets_velocity.numel()

        loss = loss_note + loss_offset + loss_duration + loss_velocity

        # Backward pass and optimize
        loss.backward()
        if clip_value is not None:  # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        if scheduler is not None:
            scheduler.step(loss)

        running_loss += loss.item() * inputs.size(0)

    accuracy = correct_predictions / total_predictions

    return running_loss / len(train_loader.dataset), accuracy

def evaluate(model, val_loader, criterion, device, batch_size):
    model.eval()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(val_loader):
            hidden = model.init_hidden(device, batch_size)

            inputs = inputs.to(device)
            targets_note = targets_note.to(device)
            targets_offset = targets_offset.to(device)
            targets_duration = targets_duration.to(device)
            targets_velocity = targets_velocity.to(device)

            # Forward pass
            output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)

            # Element-wise loss for each feature
            loss_note = criterion(output_note, targets_note).mean()
            loss_offset = criterion(output_offset, targets_offset).mean()
            loss_duration = criterion(output_duration, targets_duration).mean()
            loss_velocity = criterion(output_velocity, targets_velocity).mean()

            # Calculate accuracy
            _, predicted_notes = torch.max(output_note.data, 1)
            _, predicted_offsets = torch.max(output_offset.data, 1)
            _, predicted_durations = torch.max(output_duration.data, 1)
            _, predicted_velocities = torch.max(output_velocity.data, 1)

            correct_predictions += (predicted_notes == targets_note).sum().item()
            correct_predictions += (predicted_offsets == targets_offset).sum().item()
            correct_predictions += (predicted_durations == targets_duration).sum().item()
            correct_predictions += (predicted_velocities == targets_velocity).sum().item()

            total_predictions += targets_note.numel()
            total_predictions += targets_offset.numel()
            total_predictions += targets_duration.numel()
            total_predictions += targets_velocity.numel()

            loss = loss_note + loss_offset + loss_duration + loss_velocity

            running_loss += loss.item() * inputs.size(0)

    accuracy = correct_predictions / total_predictions

    return running_loss / len(val_loader.dataset), accuracy