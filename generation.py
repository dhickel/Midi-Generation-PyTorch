import numpy as np
import torch
from music21 import instrument, note, stream, tempo, chord, duration
import torch.nn.functional as F


def generate_midi(model, note_data, network_data, output_file='output.mid', temperature=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    prediction_output = generate_notes(model, note_data, network_data, device, temperature=temperature)
    create_midi_track(prediction_output, output_file)


def generate_notes(model, note_data, network_data, device, seq_length=600, seed_sequence=None, temperature=0.6):
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
            prediction_note, prediction_offset, prediction_duration, prediction_velocity = model(pattern)

        note_index = torch.multinomial(F.softmax(prediction_note / temperature, dim=1), 1)
        offset_index = torch.multinomial(F.softmax(prediction_offset / temperature, dim=1), 1)
        duration_index = torch.multinomial(F.softmax(prediction_duration / temperature, dim=1), 1)
        velocity_index = torch.multinomial(F.softmax(prediction_velocity / temperature, dim=1), 1)

        result = (note_data.note_table[note_index[0, 0].item()],
                  note_data.offset_table[offset_index[0, 0].item()],
                  note_data.duration_table[duration_index[0, 0].item()],
                  note_data.velocity_table[velocity_index[0, 0].item()])
        prediction_output.append(result)

        next_input = torch.tensor(
            [[[note_index.item(), offset_index.item(), duration_index.item(), velocity_index.item()]]],
            dtype=torch.float16).to(device)

        pattern = torch.cat((pattern[:, 1:, :], next_input), dim=1)

    return prediction_output


def create_midi_track(prediction_output, output_file='output.mid'):
    output_notes = []
    total_offset = -1

    last_notes = []
    last_offset = -1
    last_chord_offset = -1

    inst = instrument.Instrument()
    inst.midiProgram = 81

    # last_<x> is used to keep track of already playing notes as to not play them over each other in-humanly in edge
    # cases
    for pattern, offset, duration_value, velocity_value in prediction_output:
        if offset > last_offset:
            last_offset = offset
            last_notes.clear()

        if total_offset == -1:
            placement_offset = 0
        else:
            placement_offset = total_offset + offset

        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            notes = []

            if last_chord_offset == offset:
                continue
            else:
                last_chord_offset = offset

            for current_note in notes_in_chord:
                if (current_note in last_notes) and offset == last_offset:
                    continue
                else:
                    last_notes.append(current_note)

                new_note = note.Note(current_note)
                new_note.storedInstrument = inst
                new_note.volume.velocity = velocity_value
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = placement_offset
            new_chord.duration = duration.Duration(duration_value)
            output_notes.append(new_chord)

        else:
            if (pattern in last_notes) and offset == last_offset:
                continue
            else:
                last_notes.append(pattern)

            new_note = note.Note(pattern)
            new_note.offset = placement_offset
            new_note.duration = duration.Duration(duration_value)
            new_note.storedInstrument = inst
            new_note.volume.velocity = velocity_value  # Add velocity
            output_notes.append(new_note)

        total_offset = placement_offset

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

