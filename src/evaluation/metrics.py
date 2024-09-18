from typing import Tuple, Optional
import torch
from abc import ABC, abstractmethod
import src.evaluation.eval_helper as eval

'''
Define metrics classes for loss and score computation
Metric List:
- CovarianceMetric
- AutoCorrelationMetric
- CrossCorrelationMetric
- HistogramMetric
- SignatureMetric: SigW1Metric, SigMMDMetric

'''

class Metric(ABC):

    @property
    @abstractmethod
    def name(self):
        pass 

    def measure(self,data, **kwargs):
        pass


class CovarianceMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'CovMetric' 

    def measure(self,data):
        return eval.cov_torch(self.transform(data))

class AutoCorrelationMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'AcfMetric' 

    def measure(self,data,max_lag,stationary,dim=(0, 1),symmetric=False):
        if stationary:
            return eval.acf_torch(self.transform(data),max_lag=max_lag,dim=dim)
        else:
            return eval.non_stationary_acf_torch(self.transform(data),symmetric).to(data.device)
        

class CrossCorrelationMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'CrossCorrMetric' 

    def measure(self,data,lags,dim=(0, 1)):
        return eval.cacf_torch(self.transform(data),lags,dim)
    

class MeanAbsDiffMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'MeanAbsDiffMetric' 

    def measure(self,data):
        x1, x2 = self.transform(data)
        return eval.mean_abs_diff(x1,x2)


class MMDMetric(Metric):
    def __init__(self,transform=lambda x: x):
        self.transform = transform
    
    @property
    def name(self):
        return 'MMDMetric' 

    def measure(self,data):
        x1, x2 = self.transform(data)
        return eval.mmd(x1,x2)


class ONNDMetric(Metric):

    def __init__(self,transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'ONNDMetric'

    def measure(self,data: Tuple[torch.Tensor,torch.Tensor]):
        """
        Calculates the Outgoing Nearest Neighbour Distance (ONND) to assess the diversity of the generated data
        Parameters
        ----------
        x_real: torch.tensor, [B, L, D]
        x_fake: torch.tensor, [B, L', D']

        Returns
        -------
        ONND: float
        """
        x_real, x_fake = data
        b1, t1, d1 = x_real.shape
        b2, t2, d2 = x_fake.shape
        assert t1 == t2, "Time length does not agree!"
        assert d1 == d2, "Feature dimension does not agree!"

        # Compute samplewise difference
        x_real_repeated = x_real.repeat_interleave(b2, 0)
        x_fake_repeated = x_fake.repeat([b1, 1, 1])
        samplewise_diff = x_real_repeated - x_fake_repeated
        # Compute samplewise MSE
        MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([b1, -1])
        # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
        ONND = (torch.min(MSE_X_Y, dim=1)[0]).mean()
        return ONND


class INNDMetric(Metric):

    def __init__(self, transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'INNDMetric'

    def measure(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """
        Calculates the Incoming Nearest Neighbour Distance (INND) to assess the authenticity of the generated data
        Parameters
        ----------
        x_real: torch.tensor, [B, L, D]
        x_fake: torch.tensor, [B, L', D']

        Returns
        -------
        INND: float
        """
        x_real, x_fake = data
        b1, t1, d1 = x_real.shape
        b2, t2, d2 = x_fake.shape
        assert t1 == t2, "Time length does not agree!"
        assert d1 == d2, "Feature dimension does not agree!"

        # Compute samplewise difference
        x_fake_repeated = x_fake.repeat_interleave(b1, 0)
        x_real_repeated = x_real.repeat([b2, 1, 1])
        samplewise_diff = x_real_repeated - x_fake_repeated
        # Compute samplewise MSE
        MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([b2, -1])
        # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
        INND = (torch.min(MSE_X_Y, dim=0)[0]).mean()
        return INND


class ICDMetric(Metric):

    def __init__(self, transform=lambda x: x):
        self.transform = transform

    @property
    def name(self):
        return 'INNDMetric'

    def measure(self, data: torch.Tensor):
        """
        Calculates the Intra Class Distance (ICD) to detect a potential model collapse
        Parameters
        ----------
        x_fake: torch.tensor, [B, L, D]

        Returns
        -------
        ICD: float
        """
        x_fake = data
        batch, _, _ = x_fake.shape

        # Compute samplewise difference
        x_fake_repeated_interleave = x_fake.repeat_interleave(batch, 0)
        x_fake_repeated = x_fake.repeat([batch, 1, 1])
        samplewise_diff = x_fake_repeated_interleave - x_fake_repeated
        # Compute samplewise MSE
        MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([batch, -1])
        # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
        ICD = 2 * (MSE_X_Y).sum()
        return ICD / (batch ** 2)


class VARMetric(Metric):
    def __init__(self, alpha=0.05, transform=lambda x: x):
        self.transform = transform
        self.alpha = alpha

    @property
    def name(self):
        return 'VARMetric'

    def measure(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """
        Calculates the alpha-value at risk to assess the tail distribution match of the generated data
        Parameters
        ----------
        x_real: torch.tensor, [B, L, D]
        x_fake: torch.tensor, [B, L', D']

        Returns
        -------
        INND: float
        """
        x_fake = data
        batch, _, _ = x_fake.shape

        # Compute samplewise difference
        x_fake_repeated_interleave = x_fake.repeat_interleave(batch, 0)
        x_fake_repeated = x_fake.repeat([batch, 1, 1])
        samplewise_diff = x_fake_repeated_interleave - x_fake_repeated
        # Compute samplewise MSE
        MSE_X_Y = torch.norm(samplewise_diff, dim=2).mean(dim=1).reshape([batch, -1])
        # For every sample in x_real, compute the minimum MSE and calculate the average among all the minimums
        ICD = 2 * (MSE_X_Y).sum()
        return ICD / (batch ** 2)