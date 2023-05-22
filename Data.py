import torch
from torch.utils.data import Dataset


class NoteData:
    def __init__(self):
        self.note_table = []
        self.duration_table = []
        self.offset_table = []
        self.note_max = -1
        self.offset_max = -1
        self.training_notes = []

    @staticmethod
    def contains(lst, item):
        for i in range(0, len(lst)):
            if lst[i] == item:
                return i

    def n_vocab(self):
        return len(self.note_table)

    def d_vocab(self):
        return len(self.duration_table)

    def o_vocab(self):
        return len(self.offset_table)

    def calc_max(self):
        self.note_max = max(self.note_table)
        self.offset_max = max(self.offset_table)

    def add_note_if_absent(self, note):
        idx = self.contains(self.note_table, note)
        if idx is None:
            self.note_table.append(note)
            return len(self.note_table)
        else:
            return idx

    def add_durr_if_absent(self, duration):
        idx = self.contains(self.duration_table, duration)
        if idx is None:
            self.duration_table.append(duration)
            return len(self.duration_table)
        else:
            return idx

    def add_offs_if_absent(self, offset):
        idx = self.contains(self.offset_table, offset)
        if idx is None:
            self.offset_table.append(offset)
            return len(self.offset_table)
        else:
            return idx



class MusicDataset(Dataset):
    def __init__(self, network_data):
        print("input tensor:", network_data.input.shape)
        self.network_input = network_data.input
        self.network_output_notes = network_data.output_notes
        self.network_output_offsets = network_data.output_offsets
        self.network_output_durations = network_data.output_durations

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return self.network_input[idx], (self.network_output_notes[idx], self.network_output_offsets[idx], self.network_output_durations[idx])


class NetworkData():
    def __init__(self, network_input, network_output_notes,network_output_offsets,network_output_durations):
        self.input = network_input
        self.output_notes = network_output_notes
        self.output_offsets = network_output_offsets
        self.output_durations = network_output_durations

