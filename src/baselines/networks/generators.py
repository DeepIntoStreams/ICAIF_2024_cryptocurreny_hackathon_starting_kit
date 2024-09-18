import torch
import torch.nn as nn
from src.utils import init_weights



class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    # @abstractmethod
    def forward_(self, batch_size: int, n_lags: int, device: str):
        """ Implement here generation scheme. """
        # ...
        pass


class ConditionalLSTMGenerator(GeneratorBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_layers: int):
        super(ConditionalLSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.linear.apply(init_weights)
        # neural network to initialise h0 from the LSTM
        # we put a tanh at the end because we are initialising h0 from the LSTM, that needs to take values between [-1,1]


    def forward(self, batch_size: int, condition: torch.Tensor, n_lags: int, device: str) -> torch.Tensor:
        z = (0.1 * torch.randn(batch_size, n_lags,
                               self.input_dim - condition.shape[-1])).to(device)  # cumsum(1)
        z[:, 0, :] *= 0  # first point is fixed
        z = z.cumsum(1)
        z = torch.cat([z, condition.unsqueeze(1).repeat((1, n_lags, 1))], dim=2)

        h0 = torch.zeros(self.rnn.num_layers, batch_size,
                         self.rnn.hidden_size).to(device)

        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)

        assert x.shape[1] == n_lags
        return x


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.n_lags = config.n_lags
        self.noise_dim = config.G_input_dim
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
