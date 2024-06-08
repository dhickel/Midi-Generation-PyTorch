import glob
import os

from tqdm import tqdm

from music21 import converter, instrument, note, chord, stream, duration, pitch
from data import NoteData, MidiDataset, NetworkData
from generation import create_midi_track
from model import *
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


def get_notes_single(directory, get_flat=True):
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

        prev_event_offset = -(data.get_random_off())
        notes = []
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

                if isinstance(event, note.Note):
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
                        off_val = data.add_offs_if_absent(offset if i == 0 else 0)
                        tup = (note_val, off_val, durr_val, vel_val)
                        if not any(item is None for item in tup):
                            notes.append(tup)

                prev_event_offset = event.offset

        data.training_notes.append(notes)
    data.calc_vocab()
    return data





def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val) * 2 - 1



def prepare_sequences(note_data, device=torch.device("cuda"), sequence_length=128):
    network_input = []
    network_output_notes = []
    network_output_offsets = []
    network_output_durations = []
    network_output_velocities = []

    min_val = 0  # Assuming the minimum value in your data
    max_val = 127  # Assuming the maximum value in your data for MIDI notes

    # Create input sequences and the corresponding outputs
    for notes in note_data.training_notes:
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            # Normalize input data
            normalized_sequence_in = [
                [normalize_data(x[0], 0, 127),
                 normalize_data(x[1], 0, 1538),
                 normalize_data(x[2], 0, 1538),
                 normalize_data(x[3], 0, 127)]
                for x in sequence_in
            ]
            network_input.append(normalized_sequence_in)
            network_output_notes.append(sequence_out[0])
            network_output_offsets.append(sequence_out[1])
            network_output_durations.append(sequence_out[2])
            network_output_velocities.append(sequence_out[3])

    network_input = torch.tensor(network_input, dtype=torch.float16).to(device)

    # Shape for cross_entropy: (N) where N is batch size.
    network_output_notes = torch.tensor(network_output_notes, dtype=torch.long).view(-1).to(device)
    network_output_offsets = torch.tensor(network_output_offsets, dtype=torch.long).view(-1).to(device)
    network_output_durations = torch.tensor(network_output_durations, dtype=torch.long).view(-1).to(device)
    network_output_velocities = torch.tensor(network_output_velocities, dtype=torch.long).view(-1).to(device)

    return NetworkData(network_input, network_output_notes, network_output_offsets, network_output_durations,
                       network_output_velocities)


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

def train(model, train_loader, criterion, optimizer, device, note_data, scaler, batch_size, scheduler=None, clip_value=None):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    hidden = model.init_hidden(device, batch_size=batch_size)
    for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
        #hidden = model.detach_hidden(hidden)
        model.init_hidden(device, batch_size=batch_size)

        inputs = inputs.to(device)
        targets_note = targets_note.to(device)
        targets_offset = targets_offset.to(device)
        targets_duration = targets_duration.to(device)
        targets_velocity = targets_velocity.to(device)

        # print(inputs.shape)
        # print(targets_note.shape)
        # Forward pass
        with autocast():
            output_note, output_offset, output_duration, output_velocity, _ = model(inputs, hidden)

            # Calculate loss
            loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long())
            loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long())
            loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long())
            loss_velocity = criterion(output_velocity.view(-1, note_data.v_vocab), targets_velocity.view(-1).long())

            loss = loss_note + loss_offset + loss_duration + loss_velocity

        # Backward pass and optimization
        optimizer.zero_grad()

        scaler.scale(loss).backward()

        # Gradient clip
        if clip_value is not None:  # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        scaler.step(optimizer)
        scaler.update()

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

    accuracy = correct_predictions / (total_predictions * 4 )

    return (running_loss / len(train_loader.dataset)) / 4, accuracy


def train(model, train_loader, criterion, optimizer, device, note_data, scaler, batch_size, loss_weights,
          scheduler=None, clip_value=None):
    model.train()
    running_loss = 0.0
    running_loss_note = 0.0
    running_loss_offset = 0.0
    running_loss_duration = 0.0
    running_loss_velocity = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(train_loader):
        # Initialize and detach hidden state
        hidden = model.init_hidden(device, batch_size=batch_size)
        hidden = model.detach_hidden(hidden)

        inputs = inputs.to(device)
        targets_note = targets_note.to(device)
        targets_offset = targets_offset.to(device)
        targets_duration = targets_duration.to(device)
        targets_velocity = targets_velocity.to(device)

        optimizer.zero_grad()

        # Forward pass with autocast for mixed precision
        with autocast():
            output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)

            # Calculate individual losses
            loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long()) * loss_weights[
                'note']
            loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long()) * \
                          loss_weights['offset']
            loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long()) * \
                            loss_weights['duration']
            loss_velocity = criterion(output_velocity.view(-1, note_data.v_vocab), targets_velocity.view(-1).long()) * \
                            loss_weights['velocity']

            # Total loss is the sum of all individual weighted losses
            loss = loss_note + loss_offset + loss_duration + loss_velocity

        # Backward pass and optimization
        scaler.scale(loss).backward()

        # Gradient clipping
        if clip_value is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        running_loss_note += loss_note.item() * inputs.size(0)
        running_loss_offset += loss_offset.item() * inputs.size(0)
        running_loss_duration += loss_duration.item() * inputs.size(0)
        running_loss_velocity += loss_velocity.item() * inputs.size(0)

        # Calculate accuracy for each output
        _, predicted_notes = torch.max(output_note, 1)
        _, predicted_offsets = torch.max(output_offset, 1)
        _, predicted_durations = torch.max(output_duration, 1)
        _, predicted_velocities = torch.max(output_velocity, 1)

        total_predictions += targets_note.size(0)  # Total samples per batch
        correct_predictions += (predicted_notes == targets_note).sum().item()
        correct_predictions += (predicted_offsets == targets_offset).sum().item()
        correct_predictions += (predicted_durations == targets_duration).sum().item()
        correct_predictions += (predicted_velocities == targets_velocity).sum().item()

    # Calculate average loss and accuracy
    avg_loss = running_loss / (len(train_loader.dataset) * 4)
    avg_loss_note = running_loss_note / len(train_loader.dataset)
    avg_loss_offset = running_loss_offset / len(train_loader.dataset)
    avg_loss_duration = running_loss_duration / len(train_loader.dataset)
    avg_loss_velocity = running_loss_velocity / len(train_loader.dataset)
    accuracy = correct_predictions / (total_predictions * 4)

    print(
        f"Loss Note: {avg_loss_note:.4f}, Loss Offset: {avg_loss_offset:.4f}, Loss Duration: {avg_loss_duration:.4f}, Loss Velocity: {avg_loss_velocity:.4f}")

    return avg_loss, accuracy

def validate(model, valid_loader, criterion, device, note_data, batch_size):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        hidden = model.init_hidden(device, batch_size=batch_size)
        for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in tqdm(valid_loader):
            inputs = inputs.to(device)
            targets_note = targets_note.to(device)
            targets_offset = targets_offset.to(device)
            targets_duration = targets_duration.to(device)
            targets_velocity = targets_velocity.to(device)

            # Forward pass
            with autocast():
                output_note, output_offset, output_duration, output_velocity, hidden = model(inputs, hidden)

                # Calculate loss
                loss_note = criterion(output_note.view(-1, note_data.n_vocab), targets_note.view(-1).long())
                loss_offset = criterion(output_offset.view(-1, note_data.o_vocab), targets_offset.view(-1).long())
                loss_duration = criterion(output_duration.view(-1, note_data.d_vocab), targets_duration.view(-1).long())
                loss_velocity = criterion(output_velocity.view(-1, note_data.v_vocab), targets_velocity.view(-1).long())

                loss = loss_note + loss_offset + loss_duration + loss_velocity

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

        return (running_loss / len(valid_loader.dataset)) / 4, accuracy
