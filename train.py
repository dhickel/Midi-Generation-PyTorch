import glob
from music21 import converter, instrument, note, chord, stream, duration, pitch
from data import NoteData, MidiDataset, NetworkData
from model import *


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


def prepare_sequences(note_data, sequence_length=64):
    sequence_length = sequence_length

    network_input = []
    network_output_notes = []
    network_output_offsets = []
    network_output_durations = []
    network_output_velocities = []

    # create input sequences and the corresponding outputs
    for notes in note_data.training_notes:
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([[x[0], x[1], x[2], x[3]] for x in sequence_in])
            network_output_notes.append(sequence_out[0])
            network_output_offsets.append(sequence_out[1])
            network_output_durations.append(sequence_out[2])
            network_output_velocities.append(sequence_out[3])

    network_input = torch.tensor(network_input, dtype=torch.float16)

    # Shape for cross_entropy: (N) where N is batch size.
    network_output_notes = torch.tensor(network_output_notes, dtype=torch.long).view(-1)
    network_output_offsets = torch.tensor(network_output_offsets, dtype=torch.long).view(-1)
    network_output_durations = torch.tensor(network_output_durations, dtype=torch.long).view(-1)
    network_output_velocities = torch.tensor(network_output_velocities, dtype=torch.long).view(-1)

    return NetworkData(network_input, network_output_notes, network_output_offsets, network_output_durations,
                       network_output_velocities)


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


def train(model, train_loader, criterion, optimizer, scheduler, device, note_data, clip_value=None):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in train_loader:
        inputs = inputs.to(device)
        targets_note = targets_note.to(device)
        targets_offset = targets_offset.to(device)
        targets_duration = targets_duration.to(device)
        targets_velocity = targets_velocity.to(device)

        # Forward pass
        output_note, output_offset, output_duration, output_velocity = model(inputs)

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


def validate(model, valid_loader, criterion, device, note_data):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, (targets_note, targets_offset, targets_duration, targets_velocity) in valid_loader:
            inputs = inputs.to(device)
            targets_note = targets_note.to(device)
            targets_offset = targets_offset.to(device)
            targets_duration = targets_duration.to(device)
            targets_velocity = targets_velocity.to(device)

            # Forward pass
            output_note, output_offset, output_duration, output_velocity = model(inputs)

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
