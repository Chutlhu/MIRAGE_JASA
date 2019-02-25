import numpy as np
import torch

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size = 201, layer_size = [100, 50, 20], drop_out = 0.3):
        super(AutoEncoder, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(input_size, layer_size[0]),
            torch.nn.ReLU(True),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(layer_size[0], layer_size[1]),
            torch.nn.ReLU(True),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(layer_size[1], layer_size[2]),
            torch.nn.ReLU(True),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(layer_size[2], layer_size[1]),
            torch.nn.ReLU(True),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(layer_size[1], layer_size[0]),
            torch.nn.ReLU(True),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(layer_size[0], input_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
