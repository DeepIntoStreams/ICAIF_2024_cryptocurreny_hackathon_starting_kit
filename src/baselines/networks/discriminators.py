from torch import nn
from src.utils import deterministic_NeuralSort

from src.evaluation.strategies import *
equal_weight = EqualWeightPortfolioStrategy()
strategy = MeanReversionStrategy()

R_shape = (3,)
class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, out_dim=1):
        super(LSTMDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_size=input_dim+1,
                            hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, condition.unsqueeze(1).repeat((1, x.shape[1], 1))], dim=2)
        h = self.lstm(z)[0][:, -1:]
        x = self.linear(h)
        return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.W = config.W
        self.project = config.project
        self.alphas = config.alphas
        self.model = nn.Sequential(
            nn.Linear(3, 256), # nn.Linear(3, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2 * len(config.alphas)),
        )
        self.model_pnl = nn.Sequential(
            nn.Linear(13, 256), # nn.Linear(3, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2 * len(config.alphas)),
        )

    def project_op(self, validity):
        for i, alpha in enumerate(self.alphas):
            v = validity[:, 2*i].clone()
            e = validity[:, 2*i+1].clone()
            indicator = torch.sign(torch.as_tensor(0.5 - alpha))
            validity[:, 2*i] = indicator * ((self.W * v < e).float() * v + (self.W * v >= e).float() * (v + self.W * e) / (1 + self.W ** 2))
            validity[:, 2*i+1] = indicator * ((self.W * v < e).float() * e + (self.W * v >= e).float() * self.W * (v + self.W * e) / (1 + self.W ** 2))
        return validity


    def forward(self, x):
        PNL = strategy.get_pnl_trajectory(x) 
        PNL_s = PNL.reshape(*PNL.shape, 1).to(self.config.device)
        perm_matrix = deterministic_NeuralSort(PNL_s, self.config.temp)
        PNL_sort = torch.bmm(perm_matrix, PNL_s)
        batch_size, seq_len, _ = PNL_s.shape
        PNL_validity = self.model_pnl(PNL_sort.view(batch_size, -1))

        return PNL, PNL_validity
    
    
