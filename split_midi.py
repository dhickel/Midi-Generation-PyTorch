from music21 import converter, stream, instrument
import os

# Splits midi tracks into individual files
def extract_tracks(input_directory, output_directory):
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop over all MIDI files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.mid'):
            try:
                # Load the MIDI file
                midi = converter.parse(os.path.join(input_directory, filename))
            except Exception as e:
                print(f"Skipping file {filename} due to an error: {str(e)}")
                continue

            # Create a new directory for the extracted tracks
            new_dir = os.path.join(output_directory, filename[:-4])  # remove '.mid' from filename
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            # Loop over all tracks in the MIDI file
            for i, part in enumerate(midi.parts):
                # Create a new MIDI stream
                midi_stream = stream.Stream()
                for element in part:
                    midi_stream.append(element)

                # Save this track as a separate MIDI file
                midi_stream.write('midi', fp=os.path.join(new_dir, f'{filename[:-4]}_track_{i}.mid'))


input_directory = 'data/split'
output_directory = 'data/split2'
extract_tracks(input_directory, output_directory)