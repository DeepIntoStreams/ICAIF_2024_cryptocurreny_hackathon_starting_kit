from src.evaluation.loss import *
from src.evaluation.scores import get_discriminative_score, get_predictive_score
from src.evaluation.strategies import *
import pandas as pd
from src.utils import *
from tqdm import tqdm


STRATEGIES ={'equal_weight': EqualWeightPortfolioStrategy,
             'mean_reversion': MeanReversionStrategy,
             'trend_following': TrendFollowingStrategy,
             'vol_trading': VolatilityTradingStrategy}

def full_evaluation(fake_dataset, real_dataset, config, **kwargs):
    ec = EvaluationComponent(config, fake_dataset, real_dataset, **kwargs)
    summary_dict = ec.eval_summary()
    return summary_dict


class EvaluationComponent(object):
    '''
    Evaluation component for evaluation metrics according to config
    '''

    def __init__(self, config, fake_dataset, real_data, **kwargs):
        self.config = config
        self.fake_data = fake_dataset
        self.kwargs = kwargs
        self.n_eval = self.config.Evaluation.n_eval

        if 'seed' in kwargs:
            self.seed = kwargs['seed']
        elif 'seed' in config:
            self.seed = config.seed
        else:
            self.seed = None

        self.real_data = real_data
        self.dim = self.real_data.shape[-1]

        self.sample_size = min(self.real_data.shape[0], self.config.Evaluation.batch_size)
        print(self.sample_size)
        # set_seed(self.config.seed)
        self.data_set = self.get_data(n=self.n_eval)

        strat_name = kwargs.get('strat_name', 'equal_weight')
        self.strat = STRATEGIES[strat_name]()
        self.metrics_group = {
            'stylized_fact_scores': ['hist_loss', 'cross_corr', 'cov_loss', 'acf_loss'],
            'implicit_scores': ['discriminative_score', 'predictive_score', 'predictive_FID'],
            'sig_scores': ['sigw1', 'sig_mmd'],
            'permutation_test': ['permutation_test'],
            'distance_based_metrics': ['onnd', 'innd', 'icd'],
            'tail_scores': ['var', 'es'],
            'trading_strat_scores': ['max_drawback', 'cumulative_pnl']
        }

    def get_data(self, n=1):
        batch_size = int(self.sample_size)

        idx_all = torch.randint(self.real_data.shape[0], (batch_size * n,))
        idx_all_test = torch.randint(self.fake_data.shape[0], (batch_size * n,))
        data = {}
        for i in range(n):
            idx = idx_all[i * batch_size:(i + 1) * batch_size]
            # idx = torch.randint(real_data.shape[0], (sample_size,))
            real = self.real_data[idx]
            idx = idx_all_test[i * batch_size:(i + 1) * batch_size]
            fake = self.fake_data[idx]
            data.update({i:
                {
                    'real': real,
                    'fake': fake
                }
            })
        return data

    def eval_summary(self):

        metrics = self.config.Evaluation.metrics_enabled
        # init
        scores = {metric: [] for metric in metrics}
        summary = {}

        for grp in self.metrics_group.keys():

            metrics_in_group = [m for m in metrics if m in self.metrics_group[grp]]

            if len(metrics_in_group):

                for metric in metrics_in_group:
                    print(f'---- evaluation metric = {metric} in group = {grp} ----')

                    # create eval function by metric name
                    eval_func = getattr(self, metric)

                    if grp == 'permutation_test':
                        power, type1_error = eval_func()
                        summary['permutation_test_power'] = power
                        summary['permutation_test_type1_error'] = type1_error

                    else:
                        for i in tqdm(range(self.n_eval)):
                            real = self.data_set[i]['real']
                            fake = self.data_set[i]['fake']

                            if grp in ['stylized_fact_scores', 'sig_scores', 'distance_based_metrics', 'tail_scores',
                                       'trading_strat_scores']:

                                pnl_real = self.strat.get_pnl_trajectory(real)
                                pnl_fake = self.strat.get_pnl_trajectory(fake)
                                score = eval_func(pnl_real, pnl_fake)

                            else:
                                raise NotImplementedError(
                                    f"metric {metric} not specified in any group {self.metrics_group.keys()}")

                            # print(metric, score)
                            # update scores
                            ss = scores[metric]
                            ss.append(score)
                            scores.update({metric: ss})

                        m_mean, m_std = np.array(scores[metric]).mean(), np.array(scores[metric]).std()
                        summary[f'{metric}_mean'] = m_mean
                        summary[f'{metric}_std'] = m_std
            else:
                print(f' No metrics enabled in group = {grp}')

        df = pd.DataFrame([summary])

        return summary

    def discriminative_score(self, real_train_dl, real_test_dl, fake_train_dl, fake_test_dl):
        ecfg = self.config.Evaluation.TestMetrics.discriminative_score
        d_score_mean, _ = get_discriminative_score(
            real_train_dl, real_test_dl, fake_train_dl, fake_test_dl,
            self.config)
        return d_score_mean

    def predictive_score(self, real_train_dl, real_test_dl, fake_train_dl, fake_test_dl):
        ecfg = self.config.Evaluation.TestMetrics.predictive_score
        p_score_mean, _ = get_predictive_score(
            real_train_dl, real_test_dl, fake_train_dl, fake_test_dl,
            self.config)
        return p_score_mean

    # def sigw1(self, real, fake):
    #     ecfg = self.config.Evaluation.TestMetrics.sigw1_loss
    #     loss = to_numpy(SigW1Loss(x_real=real, depth=ecfg.depth, name='sigw1', normalise=ecfg.normalise)(fake))
    #     return loss
    #
    # def sig_mmd(self, real, fake):
    #     ecfg = self.config.Evaluation.TestMetrics.sig_mmd
    #     if False:
    #         metric = SigMMDMetric()
    #         sig_mmd = metric.measure((real, fake), depth=ecfg.depth, seed=self.seed)
    #     loss = to_numpy(SigMMDLoss(x_real=real, depth=ecfg.depth, seed=self.seed, name='sigmmd')(fake))
    #     return loss

    def cross_corr(self, real, fake):
        cross_corr = to_numpy(CrossCorrelLoss(real, name='cross_corr')(fake))
        return cross_corr

    def hist_loss(self, real, fake):
        ecfg = self.config.Evaluation.TestMetrics.hist_loss
        if ecfg.keep_init:
            loss = to_numpy(HistoLoss(real[:, 1:, :], n_bins=ecfg.n_bins, name='hist_loss')(fake[:, 1:, :]))
        else:
            loss = to_numpy(HistoLoss(real, n_bins=ecfg.n_bins, name='hist_loss')(fake))
        return loss

    def acf_loss(self, real, fake):
        ecfg = self.config.Evaluation.TestMetrics.acf_loss
        if ecfg.keep_init:
            loss = to_numpy(ACFLoss(real, name='acf_loss', stationary=ecfg.stationary)(fake))
        else:
            loss = to_numpy(ACFLoss(real[:,1:], name='acf_loss', stationary=ecfg.stationary)(fake[:,1:]))
        return loss

    def cov_loss(self, real, fake):
        loss = to_numpy(CovLoss(real, name='cov_loss')(fake))
        return loss

    # def permutation_test(self):
    #     ecfg = self.config.Evaluation.TestMetrics.permutation_test
    #     if 'recovery' in self.kwargs:
    #         recovery = self.kwargs['recovery']
    #         kwargs = {'recovery': recovery}
    #     else:
    #         kwargs = {}
    #     fake_data = loader_to_tensor(
    #         fake_loader(
    #             self.generator,
    #             num_samples=int(self.real_data.shape[0] // 2),
    #             n_lags=self.config.n_lags,
    #             batch_size=self.config.batch_size,
    #             algo=self.algo,
    #             **kwargs
    #         )
    #     )
    #     power, type1_error = sig_mmd_permutation_test(self.real_data, fake_data, ecfg.n_permutation)
    #     return power, type1_error

    def onnd(self, real, fake):
        # ecfg = self.config.Evaluation.TestMetrics.onnd
        metric = ONNDMetric()
        if real.shape[0]>8000:
            real = real[:8000]
            fake = fake[:8000]
        loss = to_numpy(metric.measure((real, fake)))
        return loss

    def innd(self, real, fake):
        # ecfg = self.config.Evaluation.TestMetrics.innd
        metric = INNDMetric()
        if real.shape[0]>8000:
            real = real[:8000]
            fake = fake[:8000]
        loss = to_numpy(metric.measure((real, fake)))
        return loss

    def icd(self, real, fake):
        # ecfg = self.config.Evaluation.TestMetrics.icd
        metric = ICDMetric()
        if fake.shape[0]>8000:
            fake = fake[:8000]
        loss = to_numpy(metric.measure(fake))
        return loss

    def max_drawback(self, real, fake):
        loss = to_numpy(MaxDrawbackLoss(real, name='max_drawback_loss')(fake))
        return loss

    def cumulative_pnl(self, real, fake):
        loss = to_numpy(CumulativePnLLoss(real, name='cum_pnl_loss')(fake))
        return loss

    def var(self, real, fake):
        ecfg = self.config.Evaluation.TestMetrics.var
        loss = to_numpy(VARLoss(real.unsqueeze(2), name='var_loss', alpha=ecfg.alpha)(fake.unsqueeze(2)))
        return loss

    def es(self, real, fake):
        ecfg = self.config.Evaluation.TestMetrics.es
        loss = to_numpy(ESLoss(real.unsqueeze(2), name='es_loss', alpha=ecfg.alpha)(fake.unsqueeze(2)))
        return loss
