import os
import pretty_midi
import numpy as np
from sklearn.model_selection import train_test_split

class MidiPreprocessor:
    def __init__(self, min_pitch=21, max_pitch=108, max_velocity=127, max_duration=32):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.max_velocity = max_velocity
        self.max_duration = max_duration

    def parse_midi_file(self, file_path):
        midi_data = pretty_midi.PrettyMIDI(file_path)
        instruments_note_data = []

        for instrument in midi_data.instruments:
            note_data = []
            for note in instrument.notes:
                note_data.append([note.start, note.end, note.pitch, note.velocity])
            instruments_note_data.append(note_data)

        return instruments_note_data

    def create_note_sequences(self, note_data):
        note_sequences = []

        for note in note_data:
            duration = note[1] - note[0]
            note_sequences.append([note[2], duration, note[3]])

        return note_sequences

    def integer_encode_sequences(self, note_sequences):
        encoded_data = []

        for note in note_sequences:
            pitch = note[0] - self.min_pitch
            duration = note[1]
            velocity = note[2]
            encoded_data.append([pitch, duration, velocity])

        return np.array(encoded_data)

    def create_training_data(self, encoded_data, sequence_length=32):
        X = []
        y = []

        for i in range(len(encoded_data) - sequence_length):
            X.append(encoded_data[i:i + sequence_length])
            y.append(encoded_data[i + sequence_length])

        return np.array(X), np.array(y)

    def normalize_data(self, X, y):
        X_normalized = X.copy().astype(float)
        y_normalized = y.copy().astype(float)

        X_normalized[:, :, 0] /= self.max_pitch
        X_normalized[:, :, 1] /= self.max_duration
        X_normalized[:, :, 2] /= self.max_velocity

        y_normalized[:, 0] /= self.max_pitch
        y_normalized[:, 1] /= self.max_duration
        y_normalized[:, 2] /= self.max_velocity

        return X_normalized, y_normalized

    # def preprocess_midi_files(self, input_directory, sequence_length=32, test_size=0.2):
    #     all_X = []
    #     all_y = []
    #
    #     for file_name in os.listdir(input_directory):
    #         file_path = os.path.join(input_directory, file_name)
    #         instruments_note_data = self.parse_midi_file(file_path)
    #
    #         for note_data in instruments_note_data:
    #             note_sequences = self.create_note_sequences(note_data)
    #             encoded_data = self.integer_encode_sequences(note_sequences)
    #             X, y = self.create_training_data(encoded_data, sequence_length)
    #             all_X.append(X)
    #             all_y.append(y)
    #
    #     all_X = np.vstack(all_X)
    #     all_y = np.vstack(all_y)
    #
    #     X_normalized, y_normalized = self.normalize_data(all_X, all_y)
    #
    #     X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_normalized, test_size=test_size, random_state=42)
    #
    #     return X_train, X_val, y_train, y_val

    def preprocess_midi_files(self, input_directory, sequence_length=32, test_size=0.2):
        all_X = []
        all_y = []

        for file_name in os.listdir(input_directory):
            file_path = os.path.join(input_directory, file_name)
            instruments_note_data = self.parse_midi_file(file_path)

            for note_data in instruments_note_data:
                note_sequences = self.create_note_sequences(note_data)
                encoded_data = self.integer_encode_sequences(note_sequences)
                X, y = self.create_training_data(encoded_data, sequence_length)

                # Check if X and y have the expected dimensions before appending
                if X.ndim == 3 and y.ndim == 2:
                    all_X.append(X)
                    all_y.append(y)

        all_X = np.vstack(all_X)
        all_y = np.vstack(all_y)

        X_normalized, y_normalized = self.normalize_data(all_X, all_y)

        X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_normalized, test_size=test_size,
                                                          random_state=42)

        return X_train, X_val, y_train, y_val


