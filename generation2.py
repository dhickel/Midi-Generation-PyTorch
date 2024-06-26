import pickle
import random

import numpy as np
import torch
from music21 import instrument, note, stream, tempo, chord, duration, interval, key, pitch
import torch.nn.functional as F


def generate_midi(model, note_data, network_data, output_file='output.mid', seed=None, temperature=1.0,
                  seq_len=1200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    prediction_output = generate_notes(model, note_data, network_data, device, seed=seed, temperature=temperature,
                                       seq_length=seq_len)
    create_midi_track(prediction_output, output_file=output_file)


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


def add_gaussian_noise(original, amount):
    # Define the lower and upper bounds for each column
    lower_bounds = torch.min(original, dim=0)[0]
    # upper_bounds = torch.max(original, dim=0)[0]
    upper_bounds = torch.tensor([0, 0, 0, 0]).to(original.device)
    # Define the standard deviations for the Gaussian noise for each column
    std_devs = (upper_bounds - lower_bounds) * amount

    # Move std_devs to the same device as original
    std_devs = std_devs.to(original.device)

    # Generate Gaussian noise and add it to the original tensor
    noisy = original + torch.randn_like(original) * std_devs

    # Clamp the values to the valid range
    noisy = torch.max(torch.min(noisy, upper_bounds), lower_bounds)

    return noisy


def add_harmonic_noise(original, amount):
    # Define the lower and upper bounds for each column
    lower_bounds = torch.min(original, dim=0)[0]
    upper_bounds = torch.tensor([0, 0, 0, 0]).to(original.device)

    # Compute the range for each column
    ranges = upper_bounds - lower_bounds

    # Generate harmonic noise and scale it based on the range
    harmonic_noise = torch.sin(torch.arange(original.size(0)) * amount)  # Adjust the frequency as needed
    harmonic_noise = harmonic_noise.unsqueeze(1)  # Add a dimension to align with ranges tensor
    harmonic_noise = harmonic_noise * ranges

    # Move harmonic_noise to the same device as original
    harmonic_noise = harmonic_noise.to(original.device)

    # Add the harmonic noise to the original tensor
    noisy = original + harmonic_noise

    # Clamp the values to the valid range
    noisy = torch.max(torch.min(noisy, upper_bounds), lower_bounds)

    return noisy


def add_uniform_noise(original, range_factor):
    # Define the lower and upper bounds for each column
    lower_bounds = torch.min(original, dim=0)[0]
    upper_bounds = torch.tensor([0, 0, 0, 0]).to(original.device)

    # Compute the range for each column
    ranges = upper_bounds - lower_bounds

    # Generate uniform noise and scale it based on the range
    uniform_noise = torch.rand_like(original) * range_factor  # Adjust the range factor as needed
    uniform_noise = uniform_noise * ranges

    # Move uniform_noise to the same device as original
    uniform_noise = uniform_noise.to(original.device)

    # Add the uniform noise to the original tensor
    noisy = original + uniform_noise

    # Clamp the values to the valid range
    noisy = torch.max(torch.min(noisy, upper_bounds), lower_bounds)

    return noisy


# def generate_note_sequence(model, seed_sequence, sequence_length, top_k=0, top_p=0.0):
#     model.eval()
#
#     sequence = seed_sequence.copy()
#     state_h, state_c = model.init_state(len(seed_sequence))
#
#     for _ in range(sequence_length):
#         x_sequence = torch.tensor([[sequence[-1]]], dtype=torch.float32)
#
#         y_preds, (state_h, state_c) = model(x_sequence, (state_h, state_c))
#
#         # Apply temperature
#         y_preds = y_preds / temperature
#
#         # top-k sampling
#         if top_k > 0:
#             top_k_values, top_k_indices = torch.topk(y_preds, k=top_k, sorted=True)
#             probabilities = F.softmax(top_k_values, dim=-1)
#             next_note = torch.multinomial(probabilities, num_samples=1)
#
#             # top-k indices need to be mapped back to the original note indices
#             next_note = top_k_indices[0][next_note]
#
#         # top-p sampling
#         elif top_p > 0.0:
#             sorted_logits, sorted_indices = torch.sort(y_preds, descending=True)
#             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#
#             # Remove tokens with cumulative probability above the threshold
#             sorted_indices_to_remove = cumulative_probs > top_p
#
#             # Shift the indices to the right to keep also the first token above the threshold
#             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#             sorted_indices_to_remove[..., 0] = 0
#
#             indices_to_remove = sorted_indices[sorted_indices_to_remove]
#             y_preds[indices_to_remove] = float('-inf')
#
#             probabilities = F.softmax(y_preds, dim=-1)
#             next_note = torch.multinomial(probabilities, num_samples=1)
#
#         # plain sampling
#         else:
#             probabilities = F.softmax(y_preds, dim=-1)
#             next_note = torch.multinomial(probabilities, num_samples=1)
#
#         sequence.append(int(next_note[0][0]))
#
#     return sequence


# def generate_notes(model, note_data, network_data, device, seq_length=1200, seed=None, temperature=0.6):
#     model.eval()
#
#     # If a seed sequence is provided, use it, else choose one randomly
#     if seed is None:
#         start = np.random.randint(0, len(network_data.input) - 1)
#         pattern = network_data.input[start].unsqueeze(0).to(device)
#     else:
#         pattern = seed.unsqueeze(0).to(device)
#
#     prediction_output = []
#
#     for note_index in range(seq_length):
#         with torch.no_grad():
#             prediction_note, prediction_offset, prediction_duration, prediction_velocity = model(pattern)
#
#         print("prediction_note",prediction_note.shape)
#         # Apply softmax and temperature to model's output
#         prediction_note = F.softmax(prediction_note / temperature, dim=-1)
#         prediction_offset = F.softmax(prediction_offset / temperature, dim=-1)
#         prediction_duration = F.softmax(prediction_duration / temperature, dim=-1)
#         prediction_velocity = F.softmax(prediction_velocity / temperature, dim=-1)
#
#         # Convert predictions to actual vocab values
#         note_indices = torch.topk(prediction_note.squeeze(), k=6).indices
#         offset_indices = torch.topk(prediction_offset.squeeze(), k=6).indices
#         duration_indices = torch.topk(prediction_duration.squeeze(), k=6).indices
#         velocity_indices = torch.topk(prediction_velocity.squeeze(), k=6).indices
#
#         notes = []
#         for idx in note_indices.tolist():
#             notes.append([note_data.get_note(i) for i in idx])
#
#         offsets = []
#         for idx in offset_indices.tolist():
#             offsets.append([note_data.get_offset(i) for i in idx])
#
#         durations = []
#         for idx in duration_indices.tolist():
#             durations.append([note_data.get_duration(i) for i in idx])
#
#         velocities = []
#         for idx in velocity_indices.tolist():
#             velocities.append([note_data.get_velocity(i) for i in idx])
#
#         result = [notes, offsets, durations, velocities]
#         print(result)
#         prediction_output.append(result)
#         #
#         # next_input = torch.tensor([note_indices, offset_indices, duration_indices, velocity_indices],
#         #                           torch.long).to(device)
#         # pattern = torch.cat((pattern[:, 6:, :], next_input.unsqueeze(0)), dim=1)
#
#         # Stack predictions into a tensor with shape [4, 6]
#         next_input = torch.stack([note_indices, offset_indices, duration_indices, velocity_indices])
#
#         # Reshape next_input to match pattern's shape [1, 1, 4, 6]
#         next_input = next_input.unsqueeze(0).unsqueeze(0)
#         print("note indicies", note_indices.shape)
#         print("pattern", pattern.shape)
#         print("next_input", next_input.shape)
#         pattern = torch.cat((pattern[:, 1:, :, :], next_input), dim=1)
#         print(pattern.shape)
#     return prediction_output



# def generate_notes(model, note_data, network_data, device, seq_length=1200, seed=None, temperature=0.6):
#     model.eval()
#
#     # If a seed sequence is provided, use it, else choose one randomly
#     if seed is None:
#         start = np.random.randint(0, len(network_data.input) - 1)
#         pattern = network_data.input[start].unsqueeze(0).to(device)
#     else:
#         pattern = seed.unsqueeze(0).to(device)
#
#     prediction_output = []
#
#     for note_index in range(seq_length):
#         with torch.no_grad():
#             prediction_note, prediction_offset, prediction_duration, prediction_velocity = model(pattern)
#         # Convert predictions to actual vocab values
#         note_indices = torch.argmax(prediction_note.squeeze(), dim=-1)
#         offset_indices = torch.argmax(prediction_offset.squeeze(), dim=-1)
#         duration_indices = torch.argmax(prediction_duration.squeeze(), dim=-1)
#         velocity_indices = torch.argmax(prediction_velocity.squeeze(), dim=-1)
#
#         notes = [[note_data.get_note(idx.item()) for idx in indices] for indices in note_indices]
#         offsets = [[note_data.get_offset(idx.item()) for idx in indices] for indices in offset_indices]
#         durations = [[note_data.get_duration(idx.item()) for idx in indices] for indices in duration_indices]
#         velocities = [[note_data.get_velocity(idx.item()) for idx in indices] for indices in velocity_indices]
#
#         result = [notes, offsets, durations, velocities]
#         prediction_output.append(result)
#
#         next_input = torch.tensor([[[idx.item() for idx in indices]] for indices in result],
#                                   dtype=torch.float16).to(device)
#         pattern = torch.cat((pattern[:, 1:, :], next_input), dim=1)
#
#     return prediction_output


def generate_notes(model, note_data, network_data, device, seq_length=1200, seed=None, temperature=0.6):
    model.eval()

    # If a seed sequence is provided, use it, else choose one randomly
    if seed is None:
        start = np.random.randint(0, len(network_data.input) - 1)
        pattern = network_data.input[start].unsqueeze(0).to(device)

    else:
        pattern = seed.unsqueeze(0).to(device)

    prediction_output = []
    hidden = model.init_hidden(device, batch_size=1)
    for note_index in range(seq_length):
        with torch.no_grad():
            prediction_note, prediction_offset, prediction_duration, prediction_velocity, hidden = model(pattern, hidden)

        # Sample from the probability distribution of the predicted notes/offsets/durations/velocities
        next_input_features = []

        # arrays to hold multiple feature elements for each feature
        notes = []
        offsets = []
        durations = []
        velocities = []

        # loop over the elements of each feature
        for i in range(6):
            note_index = torch.multinomial(F.softmax(prediction_note[0, i, :] / temperature, dim=0), 1)
            offset_index = torch.multinomial(F.softmax(prediction_offset[0, i, :] / temperature, dim=0), 1)
            duration_index = torch.multinomial(F.softmax(prediction_duration[0, i, :] / temperature, dim=0), 1)
            velocity_index = torch.multinomial(F.softmax(prediction_velocity[0, i, :] / temperature, dim=0), 1)

            note = note_data.get_note(note_index.item())
            offset = note_data.get_offset(offset_index.item())
            duration = note_data.get_duration(duration_index.item())
            velocity = note_data.get_velocity(velocity_index.item())

            notes.append(note)
            offsets.append(offset)
            durations.append(duration)
            velocities.append(velocity)

            next_input_feature = [note_index.item(), offset_index.item(), duration_index.item(), velocity_index.item()]
            next_input_features.append(next_input_feature)

        result = (notes, offsets, durations, velocities)
        prediction_output.append(result)

        next_input = torch.tensor([next_input_features], dtype=torch.long).to(device)

        # Reshape next_input to match the dimensions of pattern before concatenation
        next_input = next_input.permute(0, 2, 1)
        next_input = next_input.unsqueeze(1)  # making it [1, 1, 4, 6]

        pattern = torch.cat((pattern[:, 1:, :, :], next_input), dim=1)

    return prediction_output


def generate_notes_test( note_data, inputs,  seq_length=1200, seed=None, temperature=1):


    prediction_output = []
    for input_array in inputs:
        # arrays to hold multiple feature elements for each feature
        notes = []
        offsets = []
        durations = []
        velocities = []

        for i in range(6):

            note_index = input_array[0][i]
            offset_index = input_array[1][i]
            duration_index = input_array[2][i]
            velocity_index = input_array[3][i]
            if(note_index == 0):
                continue

            note = note_data.get_note(note_index.item())
            offset = note_data.get_offset(offset_index.item())
            duration = note_data.get_duration(duration_index.item())
            velocity = note_data.get_velocity(velocity_index.item())


            notes.append(note)
            offsets.append(offset)
            durations.append(duration)
            velocities.append(velocity)

            #
            # next_input_feature = [note_index.item(), offset_index.item(), duration_index.item(), velocity_index.item()]
            # next_input_features.append(next_input_feature)

        result = (notes, offsets, durations, velocities)


        prediction_output.append(result)
        #
        # next_input = torch.tensor([next_input_features], torch.long).to(device)
        #
        # # Reshape next_input to match the dimensions of pattern before concatenation
        # next_input = next_input.permute(0, 2, 1)
        # next_input = next_input.unsqueeze(1)  # making it [1, 1, 4, 6]
        #
        # pattern = torch.cat((pattern[:, 1:, :, :], next_input), dim=1)
    print(len(prediction_output))
    return prediction_output





# def create_midi_track(prediction_output, return_stream=False, output_file='output.mid'):
#     output_notes = []
#     total_offset = -1
#
#     last_notes = []
#     last_offset = -1
#     last_chord_offset = -1
#
#     inst = instrument.Instrument()
#     inst.midiProgram = 81
#
#     # last_<x> is used to keep track of already playing notes as to not play them over each other in-humanly in edge
#     # cases
#     for pattern, offset, duration_value, velocity_value in prediction_output:
#         if offset > last_offset:
#             last_offset = offset
#             last_notes.clear()
#
#         if total_offset == -1:
#             placement_offset = 0
#         else:
#             placement_offset = total_offset + offset
#
#         if '.' in pattern:
#             notes_in_chord = pattern.split('.')
#             notes = []
#
#             if last_chord_offset == offset:
#                 continue
#             else:
#                 last_chord_offset = offset
#
#             for current_note in notes_in_chord:
#                 if (current_note in last_notes) and offset == last_offset:
#                     continue
#                 else:
#                     last_notes.append(current_note)
#
#                 new_note = note.Note(current_note)
#                 new_note.storedInstrument = inst
#                 new_note.volume.velocity = velocity_value
#                 notes.append(new_note)
#
#             new_chord = chord.Chord(notes)
#             new_chord.offset = placement_offset
#             new_chord.duration = duration.Duration(duration_value)
#             output_notes.append(new_chord)
#
#         else:
#             if (pattern in last_notes) and offset == last_offset:
#                 continue
#             else:
#                 last_notes.append(pattern)
#
#             new_note = note.Note(pattern)
#             new_note.offset = placement_offset
#             new_note.duration = duration.Duration(duration_value)
#             new_note.storedInstrument = inst
#             new_note.volume.velocity = velocity_value  # Add velocity
#             output_notes.append(new_note)
#
#         total_offset = placement_offset
#
#     midi_stream = stream.Stream(output_notes)
#
#     if not return_stream:
#         print("here")
#         midi_stream.write('midi', fp=output_file)
#     else:
#         return midi_stream


def create_midi_track(prediction_output, return_stream=False, output_file='output.mid'):
    output_notes = []
    total_offset = -1

    last_notes = []
    last_offset = -1
    last_chord_offset = -1

    inst = instrument.Instrument()
    inst.midiProgram = 81

    # last_<x> is used to keep track of already playing notes as to not play them over each other in-humanly in edge
    # cases
    placement_offset = -1
    total_offset = 0
    for notes, offsets, durations, velocities in prediction_output:

        if total_offset == -1:
            placement_offset = 0
        else:
            placement_offset = total_offset + offsets[0]

        valid_step = False
        for i in range(len(notes)):
            if any(v is None for v in (notes[i], offsets[i], durations[i], velocities[i])):
                continue

            new_note = note.Note(midi=notes[i])
            new_note.offset = placement_offset
            new_note.duration = duration.Duration(durations[i])
            new_note.storedInstrument = inst
            new_note.volume.velocity = velocities[i]  # Add velocity
            output_notes.append(new_note)
            valid_step = True

        if valid_step:
            total_offset = placement_offset

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=output_file)
    if not return_stream:
        print("saved_track")
        midi_stream.write('midi', fp=output_file)
    else:

        return midi_stream


def create_midi_track_test(prediction_output, return_stream=False, output_file='output.mid'):
    output_notes = []
    total_offset = -1

    last_notes = []
    last_offset = -1
    last_chord_offset = -1

    inst = instrument.Instrument()
    inst.midiProgram = 81

    # last_<x> is used to keep track of already playing notes as to not play them over each other in-humanly in edge
    # cases
    placement_offset = -1
    total_offset = 0
    for notes, offsets, durations, velocities in prediction_output:

        if total_offset == -1:
            placement_offset = 0
        else:
            placement_offset = total_offset + offsets[0]

        valid_step = False
        for i in range(len(notes)):
            if any(v is None for v in (notes[i], offsets[i], durations[i], velocities[i])):
                continue


            new_note = note.Note(midi=notes[i])
            new_note.offset = placement_offset
            new_note.duration = duration.Duration(durations[i])
            new_note.storedInstrument = inst
            new_note.volume.velocity = velocities[i]  # Add velocity
            output_notes.append(new_note)
            valid_step = True

        if valid_step:
            total_offset = placement_offset

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=output_file)
    if not return_stream:
        print("saved_track")
        midi_stream.write('midi', fp=output_file)
    else:

        return midi_stream



def get_transpose_seed(note_data, seq_length, t_key, device, minor=False):
    note_ar = note_data.training_notes
    note_arr = note_ar[random.randint(0, len(note_ar))]
    while len(note_arr) < seq_length * 10:
        note_arr = note_ar[random.randint(0, len(note_ar))]

    rnd_idx = random.randint(0, len(note_arr) - (seq_length * 10 + 1))
    rnd_seq = note_arr[rnd_idx: rnd_idx + (seq_length * 10)]

    print("Seq Length before:", len(rnd_seq))
    for i in range(0, len(rnd_seq)):
        seq = rnd_seq[i]
        nnote = note_data.note_table[seq[0]]
        offset = note_data.offset_table[seq[1]]
        durr = note_data.duration_table[seq[2]]
        vel = note_data.velocity_table[seq[3]]
        rnd_seq[i] = (nnote, offset, durr, vel)

    print("Seq Length after:", len(rnd_seq))
    rnd_midi = create_midi_track(rnd_seq, return_stream=True)

    original_key = rnd_midi.analyze('key')
    if minor:
        target_key = key.Key(t_key, 'minor')
    else:
        target_key = key.Key(t_key)
    transposition_interval = interval.Interval(original_key.tonic, target_key.tonic)
    transposed_stream = rnd_midi.transpose(transposition_interval)
    notes_to_parse = transposed_stream.flat.notes

    print("notes to parse length:", len(notes_to_parse))
    notes = []
    prev_event_end = 0
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
                note_val = note_data.get_note_idx(note_val)
                durr_val = note_data.get_durr_idx(durr_val)
                offset = note_data.get_offs_idx(offset)
                vel_val = note_data.get_vel_idx(vel_val)

                if note_val is None or durr_val is None or offset is None or vel_val is None:
                    continue
                notes.append((note_val, offset, durr_val, vel_val))
                prev_event_end = event.offset
            else:
                print("none issue")

        if len(notes) == seq_length:
            break

    tensor_input = []
    print("notes length", len(notes))
    for note_tuple in notes:
        sequence_in = note_tuple[:seq_length]  # Get first seq_length elements
        tensor_input.append([x for x in sequence_in])

    # Make sure there is at least one sequence in tensor_input
    if len(tensor_input) == 0:
        raise ValueError("There are no sequences in 'tensor_input'. Check 'seq_length' and 'notes'.")

    return torch.tensor(tensor_input, dtype=torch.float16).to(device)
