"""
This is a sample file. Any user must provide a python function named init_generator() which:
    - initializes an instance of the generator,
    - loads the model parameters from model_dict.py,
    - returns the model.
"""
import numpy as np
import os
import pickle
import torch
import torch.nn as nn

print(os.path.abspath(__file__))
PATH_TO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dict.pkl')
PATH_TO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_log_return.pkl')
DEVICE = "cuda"

class Generator(nn.Module):
    def __init__(self, n_lags, G_input_dim):
        super(Generator, self).__init__()
        self.n_lags = n_lags
        self.noise_dim = G_input_dim
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.noise_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 3),
        )

    def forward(self, batch_size: int, device):
        z = (
            torch.randn(
                batch_size,
                self.n_lags,
                self.noise_dim,
            )
        ).to(device)
        res = self.model(z)
        res = torch.clamp(res, min=-1, max=1)
        return res


def init_generator():
    print("Initialisation of the model.")
    generator = Generator(n_lags=24, G_input_dim=3).to(DEVICE)
    print("Loading the model.")
    # Load from .pkl
    with open(PATH_TO_MODEL, "rb") as f:
        model_param = pickle.load(f)
    generator.load_state_dict(model_param)
    generator.eval()
    return generator


if __name__ == '__main__':
    generator = init_generator()
    print("Generator loaded. Generate fake data.")
    with torch.no_grad():
        fake_data = generator(batch_size = 1800, device = DEVICE)
    print(fake_data[0, 0:10, :])
