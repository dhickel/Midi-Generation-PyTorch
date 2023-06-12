import random
from torch.utils.data import Dataset

class NoteData:
    def __init__(self):
        self.note_table = []
        self.duration_table = []
        self.offset_table = []
        self.velocity_table = []
        self.training_notes = []
        self.n_vocab = 0
        self.o_vocab = 0
        self.d_vocab = 0
        self.v_vocab = 0
        self.rand = [0.25, 0.5, 1]

    @staticmethod
    def contains(lst, item):
        for i in range(len(lst)):
            if lst[i] == item:
                return i
        return None

    def calc_vocab(self):
        self.n_vocab = len(self.note_table) + 1
        self.o_vocab = len(self.offset_table) + 1
        self.d_vocab = len(self.duration_table) + 1
        self.v_vocab = len(self.velocity_table) + 1

    def add_note_if_absent(self, note):
        idx = self.contains(self.note_table, note)
        if idx is None:
            self.note_table.append(note)
            return len(self.note_table) - 1
        else:
            return idx

    def add_durr_if_absent(self, duration):
        durr = round(duration, 4)
        idx = self.contains(self.duration_table, durr)
        if idx is None:
            self.duration_table.append(durr)
            return len(self.duration_table) - 1
        else:
            return idx

    def add_vel_if_absent(self, velocity):
        idx = self.contains(self.velocity_table, velocity)
        if idx is None:
            self.velocity_table.append(velocity)
            return len(self.velocity_table) - 1
        else:
            return idx

    def add_offs_if_absent(self, offset):
        off = round(float(offset), 4)
        if off >= 4:
            off = 4
        idx = self.contains(self.offset_table, off)
        if idx is None:
            self.offset_table.append(off)
            return len(self.offset_table) - 1
        else:
            return idx

    def get_random_off(self):
        return self.rand[random.randint(0, len(self.rand) - 1)]


class MidiDataset(Dataset):
    def __init__(self, network_data):
        self.network_input = network_data.input
        self.network_output_notes = network_data.output_notes
        self.network_output_offsets = network_data.output_offsets
        self.network_output_durations = network_data.output_durations
        self.network_output_velocities = network_data.output_velocities

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], (
            self.network_output_notes[idx], self.network_output_offsets[idx], self.network_output_durations[idx],
            self.network_output_velocities[idx])


class NetworkData:
    def __init__(self, network_input, network_output_notes, network_output_offsets, network_output_durations,
                 network_output_velocities):
        self.input = network_input
        self.output_notes = network_output_notes
        self.output_offsets = network_output_offsets
        self.output_durations = network_output_durations
        self.output_velocities = network_output_velocities