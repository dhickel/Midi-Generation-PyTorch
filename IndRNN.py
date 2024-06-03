import torch
from torch import nn


class IdnRNN(nn.Module):
    def __init__(self, input_size, hidden_dim, batch_first):
        super(IdnRNN, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first

        self.i2h = nn.Linear(input_size, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.i2h.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.h2h.weight, nonlinearity='relu')

        if self.i2h.bias is not None:
            nn.init.zeros_(self.i2h.bias)
        if self.h2h.bias is not None:
            nn.init.zeros_(self.h2h.bias)

    def forward(self, x):
        if self.batch_first:
            x = x.transpose(0, 1)

        outputs = []
        hidden = x.new_zeros(x.size(1), self.hidden_dim)

        for i, batch in enumerate(x):
            hidden = self.i2h(batch) + self.h2h(hidden)
            hidden = self.relu(hidden)
            outputs.append(hidden.unsqueeze(0))

        output = torch.cat(outputs, 0)
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class MultiLayerIdnRNN(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, batch_first):
        super(MultiLayerIdnRNN, self).__init__()

        self.layers = nn.ModuleList([IdnRNN(input_size if i == 0 else hidden_dim,
                                            hidden_dim,
                                            batch_first) for i in range(num_layers)])

    def forward(self, x):
        hidden_states = []
        for layer in self.layers:
            x, hidden = layer(x)
            hidden_states.append(hidden)
        return x, torch.stack(hidden_states)