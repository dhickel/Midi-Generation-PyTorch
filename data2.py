import pickle
import random

import numpy
import torch
from music21 import pitch
from torch.utils.data import Dataset


class NoteData:
    def __init__(self, ppqm=None, max_durr_off_quarters=None):
        self.ppqm = 384 if ppqm is None else ppqm
        self.max_durr_off = 4 if max_durr_off_quarters is None else max_durr_off_quarters
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
        self.init()

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
            print(f"Error Note{note} Not Found!!!! This Should Not Happen!!!")
            self.note_table.append(note)
            return len(self.note_table) - 1
        else:
            return idx

    def add_durr_if_absent(self, durr):
        durr = durr
        idx = self.contains(self.duration_table, durr)
        if idx is None:
            idx = self.quantize(self.duration_table, durr)
          #  print(f"Quantized Duration:{durr} to {self.duration_table[idx]}")
        return idx

    def add_vel_if_absent(self, vel):
        idx = self.contains(self.velocity_table, vel)
        if idx is None:
            print(f"Error Note{vel} Not Found!!!! This Should Not Happen!!!")
            self.velocity_table.append(vel)
            return len(self.velocity_table) - 1
        else:
            return idx

    def add_offs_if_absent(self, offset):
        off = offset
        idx = self.contains(self.offset_table, off)
        if idx is None:
            idx = self.quantize(self.offset_table, off)
           # print(f"Quantized Off:{offset} to {self.offset_table[idx]}")
        return idx

    def get_note(self, idx):
        if idx < 0 or idx > len(self.note_table) - 1:
            return None
        else:
            return self.note_table[idx]

    def get_offset(self, idx):
        if idx < 0 or idx > len(self.offset_table) - 1:
            return None
        else:
            return self.offset_table[idx]

    def get_duration(self, idx):
        if idx < 0 or idx > len(self.duration_table) - 1:
            return None
        else:
            return self.duration_table[idx]

    def get_velocity(self, idx):
        if idx < 0 or idx > len(self.velocity_table) - 1:
            return None
        else:
            return self.velocity_table[idx]

    def get_note_idx(self, note):
        return self.contains(self.note_table, note)

    def get_durr_idx(self, durr):
        return self.contains(self.duration_table, durr)

    def get_vel_idx(self, velocity):
        return self.contains(self.velocity_table, velocity)

    def get_offs_idx(self, offset):
        return self.contains(self.offset_table, offset)

    def get_random_off(self):
        return self.rand[random.randint(0, len(self.rand) - 1)]

    @staticmethod
    def quantize(lst, value):

        for i in range (1,len(lst)):
            if lst[i] > value:
                return i-1 if (i-1) > 0 else 1

        return len(lst) - 1

    # @staticmethod
    # def quantize(lst, value):
    #     array = numpy.array(lst[1:])  # exclude padding token at index 0
    #     idx = (numpy.abs(array - value)).argmin()
    #     return idx + 1  # adjust index because we excluded the first element

    def init(self):

        self.note_table.append(None)
        self.duration_table.append(None)
        self.offset_table.append(None)
        self.offset_table.append(0)
        self.velocity_table.append(None)



        for i in range(0, 128):  # MIDI note numbers range from 0 to 127
            p = pitch.Pitch(midi=i).nameWithOctave
            print(p)
            self.note_table.append(p)
            self.velocity_table.append(i)



        for i in range(0, 3073):
            if i == 1537:
                self.ppqm = int(self.ppqm / 2)
            elif i == 2305:
                self.ppqm = int(self.ppqm / 4)

            value = i / self.ppqm
            self.offset_table.append(value)
            self.duration_table.append(value)

        print(self.offset_table)








class MidiDataset(Dataset):
    def __init__(self, network_data):
        self.network_input = network_data.input
        self.network_output_notes = network_data.output_notes
        self.network_output_offsets = network_data.output_offsets
        self.network_output_durations = network_data.output_durations
        self.network_output_velocities = network_data.output_velocities

    def __len__(self):
        return len(self.network_input)

    # def __getitem__(self, idx):
    #     return self.network_input[idx], (
    #         self.network_output_notes[idx], self.network_output_offsets[idx], self.network_output_durations[idx],
    #         self.network_output_velocities[idx])
    #
    #
    # def __getitem__(self, idx):
    #     # Get the inputs and targets
    #     inputs = self.network_input[idx]
    #     targets_notes = self.network_output_notes[idx]
    #     targets_offsets = self.network_output_offsets[idx]
    #     targets_durations = self.network_output_durations[idx]
    #     targets_velocities = self.network_output_velocities[idx]
    #
    #     # Stack the targets along a new dimension to create a single tensor
    #     targets = torch.stack((targets_notes, targets_offsets, targets_durations, targets_velocities), -1)
    #
    #     return inputs, targets

    def __getitem__(self, idx):
        # Get the inputs and targets
        inputs = self.network_input[idx]

        # Convert the targets into tensor
        targets_notes = torch.tensor(self.network_output_notes[idx])
        targets_offsets = torch.tensor(self.network_output_offsets[idx])
        targets_durations = torch.tensor(self.network_output_durations[idx])
        targets_velocities = torch.tensor(self.network_output_velocities[idx])

        # Stack the targets along a new dimension to create a single tensor
        targets = torch.stack((targets_notes, targets_offsets, targets_durations, targets_velocities), -1)

        return inputs, targets_notes, targets_offsets, targets_durations, targets_velocities


class NetworkData:
    def __init__(self, network_input, network_output_notes, network_output_offsets, network_output_durations,
                 network_output_velocities):
        self.input = network_input
        self.output_notes = network_output_notes
        self.output_offsets = network_output_offsets
        self.output_durations = network_output_durations
        self.output_velocities = network_output_velocities

print(NoteData())