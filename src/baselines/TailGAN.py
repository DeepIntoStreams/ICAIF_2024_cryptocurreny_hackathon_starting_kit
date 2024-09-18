from torch import nn
from src.baselines.base import BaseTrainer
from tqdm import tqdm
from os import path as pt
from src.utils import save_obj
from src.utils import load_config
from src.evaluation.strategies import *

config_dir = pt.join("configs/config.yaml")
config = (load_config(config_dir))


def G1(v):
    return v

def G2(e, scale=1):
    return scale * torch.exp(e / scale)

def G2in(e, scale=1):
    return scale ** 2 * torch.exp(e / scale)

def G1_quant(v, W=config.W):
    return - W * v ** 2 / 2

def G2_quant(e, alpha):
    return alpha * e

def G2in_quant(e, alpha):
    return alpha * e ** 2 / 2

def S_stats(v, e, X, alpha):
    """
    For a given quantile, here named alpha, calculate the score function value
    """
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1(v) - G1(X)) + 1. / alpha * G2(e) * (X<=v).float() * (v - X) + G2(e) * (e - v) - G2in(e)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1(X) - G1(v)) + 1. / alpha_inverse * G2(-e) * (X>=v).float() * (X - v) + G2(-e) * (v - e) - G2in(-e)
    return torch.mean(rt)

def S_quant(v, e, X, alpha, W=config.W):
    """
    For a given quantile, here named alpha, calculate the score function value
    """
    X = X.to(v.device)
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha * G2_quant(e,alpha) * (X<=v).float() * (v - X) + G2_quant(e,alpha) * (e - v) - G2in_quant(e,alpha)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha_inverse * G2_quant(-e,alpha_inverse) * (X>=v).float() * (X - v) + G2_quant(-e,alpha_inverse) * (v - e) - G2in_quant(-e,alpha_inverse)
    return torch.mean(rt)


class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.alphas = config.alphas
        self.score_name = config.score
        if self.score_name == 'quant':
            self.score_alpha = S_quant
        elif self.score_name == 'stats':
            self.score_alpha = S_stats
        else:
            self.score_alpha = None

    def forward(self, PNL_validity, PNL):
        # Score
        loss = 0
        for i, alpha in enumerate(self.alphas):
            PNL_var = PNL_validity[:, [2 * i]]
            PNL_es = PNL_validity[:, [2 * i + 1]]
            loss += self.score_alpha(PNL_var, PNL_es, PNL, alpha)

        return loss



class TailGANTrainer(BaseTrainer):
    def __init__(self, D, G, train_dl, config,
                 **kwargs):
        super(TailGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)),
            **kwargs
        )

        self.config = config
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D = D
        self.D_optimizer = torch.optim.Adam(
            D.parameters(), lr=config.lr_D, betas=(0, 0.9))  # TailGAN: lr=1e-7, betas=(0.5, 0.999)

        self.train_dl = train_dl
        self.reg_param = 0
        self.criterion = Score()

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)
    
    def step(self, device, step):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake

            with torch.no_grad():
                x_real_batch = next(iter(self.train_dl)) # .to(device) # init_price & log_return
                x_fake_log_return = self.G(self.config.batch_size, device)

            x_fake = [x_real_batch[0], x_fake_log_return] 
            init_prices_real = x_real_batch[0] 
            log_returns_real = x_real_batch[1]
            price_real = log_return_to_price(log_returns_real, init_prices_real)
            init_prices_gen = x_fake[0] 
            log_returns_gen = x_fake[1]
            price_gen = log_return_to_price(log_returns_gen, init_prices_gen)
            D_loss = self.D_trainstep(price_gen, price_real)
            if i == 0:
                self.losses_history["D_loss"].append(D_loss)
            G_loss = self.G_trainstep(price_gen, price_real, device, step)
            

    def D_trainstep(self, x_fake, x_real):
        self.D_optimizer.zero_grad()
        # Adversarial loss
        self.D = self.D.to(config.device)
        PNL, PNL_validity = self.D(x_real)
        gen_PNL, gen_PNL_validity = self.D(x_fake)
        real_score = self.criterion(PNL_validity, PNL)   
        fake_score = self.criterion(gen_PNL_validity, PNL)
        loss_D = real_score - fake_score
        # Update the Gradient in Discriminator
        loss_D.backward(retain_graph=True)
        self.D_optimizer.step()
        return loss_D.item()

    def G_trainstep(self, x_fake, x_real, device, step):
        PNL, PNL_validity = self.D(x_real)
        self.G.train()
        self.G_optimizer.zero_grad()
        self.D.train()
        gen_PNL, gen_PNL_validity = self.D(x_fake)
        loss_G = self.criterion(gen_PNL_validity, PNL)
        # Update the Gradient in Generator
        loss_G.backward(retain_graph=True)
        self.G_optimizer.step()
        return loss_G.item()
    
    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = torch.nn.BCELoss()(torch.nn.Sigmoid()(d_out), targets)
        return loss

    def save_model_dict(self):
        save_obj(self.G.state_dict(), pt.join(
            self.config.exp_dir, 'generator_state_dict.pt'))
