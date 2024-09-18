from src.evaluation.metrics import *
import numpy as np
from torch import nn

def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))
def cc_diff(x): return torch.abs(x).sum(0)
def cov_diff(x): return torch.abs(x).mean()

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x, seed=None):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo
        self.seed = seed

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()
    
    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)

class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        self.metric = AutoCorrelationMetric(self.transform)
        self.acf_calc = lambda x: self.metric.measure(x, self.max_lag, stationary,dim=(0, 1),symmetric=False)
        self.acf_real = self.acf_calc(x_real)

    def compute(self, x_fake):
        acf_fake = self.acf_calc(x_fake)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


class MeanLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean = x_real.mean((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.mean((0, 1)) - self.mean)


class StdLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.std((0, 1)) - self.std_real)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, max_lag=64, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=cc_diff, **kwargs)
        self.lags = max_lag
        self.metric = CrossCorrelationMetric(self.transform)
        self.cross_correl_real = self.metric.measure(x_real,self.lags).mean(0)[0]
        self.max_lag = max_lag

    def compute(self, x_fake):
        cross_correl_fake = self.metric.measure(x_fake,lags=self.lags).mean(0)[0]
        loss = self.norm_foo(
            cross_correl_fake - self.cross_correl_real.to(x_fake.device)).unsqueeze(0)
        return loss


# unused
class cross_correlation(Loss):
    def __init__(self, x_real, **kwargs):
        super(cross_correlation).__init__(**kwargs)
        self.x_real = x_real

    def compute(self, x_fake):
        fake_corre = torch.from_numpy(np.corrcoef(
            x_fake.mean(1).permute(1, 0))).float()
        real_corre = torch.from_numpy(np.corrcoef(
            self.x_real.mean(1).permute(1, 0))).float()
        return torch.abs(fake_corre-real_corre)


def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b+1e-5 if b == a else b
    # delta = (b - a) / n_bins
    bins = torch.linspace(a, b, n_bins+1)
    delta = bins[1]-bins[0]
    # bins = torch.arange(a, b + 1.5e-5, step=delta)
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins


class HistoLoss(Loss):

    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            # Exclude the initial point
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            tmp_loss = list()
            # Exclude the initial point
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous(
                ).view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(
                    x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(
                    density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


class CovLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CovLoss, self).__init__(norm_foo=cov_diff, **kwargs)
        self.metric = CovarianceMetric(self.transform)
        self.covariance_real = self.metric.measure(x_real)
    def compute(self, x_fake):
        covariance_fake = self.metric.measure(x_fake)
        loss = self.norm_foo(covariance_fake -
                             self.covariance_real.to(x_fake.device))
        return loss


class VARLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(VARLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='var')

    def compute(self, x_fake):
        loss = list()
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='var')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                loss.append(abs_metric)
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

class ESLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(ESLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='es')

    def compute(self, x_fake):
        loss = list()
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='es')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                loss.append(abs_metric)
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

def tail_metric(x, alpha, statistic):
    res = list()
    for i in range(x.shape[2]):
        tmp_res = list()
        # Exclude the initial point
        for t in range(x.shape[1]):
            x_ti = x[:, t, i].reshape(-1, 1)
            sorted_arr, _ = torch.sort(x_ti)
            var_alpha_index = int(alpha * len(sorted_arr))
            var_alpha = sorted_arr[var_alpha_index]
            if statistic == "es":
                es_values = sorted_arr[:var_alpha_index + 1]
                es_alpha = es_values.mean()
                tmp_res.append(es_alpha)
            else:
                tmp_res.append(var_alpha)
        res.append(tmp_res)
    return res

class MaxDrawbackLoss(Loss):
    def __init__(self, x_real, **kwargs):
        name = kwargs.pop('name')
        super(MaxDrawbackLoss, self).__init__(name=name)
        self.max_drawback = compute_max_drawdown(pnls=x_real)

    def compute(self, x_fake):
        loss = list()
        max_drawback_fake = compute_max_drawdown(pnls=x_fake)
        loss = torch.abs(self.max_drawback - max_drawback_fake)
        return loss


def compute_max_drawdown(pnls: torch.Tensor):
    """
    Compute the maximum drawdown for a batch of PnL trajectories.

    :param pnls: Tensor of shape [N, T], where N is the number of batches and T is the number of time steps.
                 This tensor represents the cumulative PnL for each batch.
    :return: Tensor of shape [N] representing the maximum drawdown for each batch.
    """
    # Compute the running maximum PnL at each time step for each batch
    running_max = torch.cummax(pnls, dim=1)[0]  # Shape [N, T], the maximum PnL at each time step

    # Compute the drawdown at each time step: (Peak PnL - Current PnL) / Peak PnL
    drawdowns = (running_max - pnls)  # Shape [N, T]

    # Compute the maximum drawdown for each batch
    max_drawdown = torch.max(drawdowns, dim=1)[0]  # Shape [N], maximum drawdown for each batch

    return max_drawdown

class CumulativePnLLoss(Loss):
    def __init__(self, x_real, **kwargs):
        name = kwargs.pop('name')
        super(CumulativePnLLoss, self).__init__(name=name)
        self.cum_pnl = x_real[:,-1]

    def compute(self, x_fake):
        cum_pnl_fake = x_fake[:,-1]
        loss = torch.abs(self.cum_pnl - cum_pnl_fake)
        return loss