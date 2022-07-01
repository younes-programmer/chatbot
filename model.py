import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_sizes, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_sizes, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.relu(output)
        output = self.layer3(output)
        return output
