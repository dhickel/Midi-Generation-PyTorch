import os
import shutil
from music21 import converter, meter


# Sorts midi files by their time signature
def sort_midi_files(input_dir, output_dir):
    # Loop through all files in the directory
    for filename in os.listdir(input_dir):
        # Check file extension
        if filename.endswith('.mid') or filename.endswith('.MID'):
            # Get the full file path
            filepath = os.path.join(input_dir, filename)

            try:
                # Convert to a music21 stream
                midi = converter.parse(filepath)

                # Get the time signature(s)
                time_signatures = midi.recurse().getElementsByClass(meter.TimeSignature)
                if time_signatures:
                    # Get the first time signature
                    ts = time_signatures[0]
                    # Prepare the directory name (e.g., "4-4")
                    ts_dir = f"{ts.numerator}-{ts.denominator}"
                    # Prepare the destination directory path
                    destination_dir = os.path.join(output_dir, ts_dir)
                    # Make sure the directory exists
                    os.makedirs(destination_dir, exist_ok=True)
                    # Prepare the destination file path
                    destination_filepath = os.path.join(destination_dir, filename)
                    # Copy the file to the new directory
                    filename.replace(".MID", ".mid")
                    shutil.copy(filepath, destination_filepath)
                    print(f"Copied {filename} to {destination_filepath}")
            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")

input_dir = "data/split"  # Replace with your input directory
output_dir = "data/time_sig"  # Replace with your output directory
sort_midi_files(input_dir, output_dir)