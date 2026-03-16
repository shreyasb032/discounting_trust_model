from math import log
from typing import Dict

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, loggamma

from classes.DiscountFactors import ConstantDF

WS = 1.0
WF = 1.0


class ParamsEstimatorBFGS:
    def __init__(self, discount_factor: ConstantDF):
        self.performance_history = []
        self.trust_feedback = []
        self.df = discount_factor

    def clear(self):
        self.performance_history.clear()
        self.trust_feedback.clear()

    def add_performance(self, performance):
        """Adds performance at sites where feedback was not queried.
        Also adds a placeholder to the trust feedback array"""
        self.performance_history.append(performance)
        self.trust_feedback.append(-1.0)

    @staticmethod
    def neg_log_likelihood(x, *args):
        """
        The negative log-likelihood function
        :param x: the trust params in order [alpha0, beta0, ws, wf]
        """
        trust_history, perf_history, _df = args
        logl = 0
        alpha0 = x[0]
        beta0 = x[1]
        ws = x[2]
        wf = x[3]

        alpha = alpha0
        beta = beta0
        for i, (t, p) in enumerate(zip(trust_history, perf_history)):
            df = _df.get_value(i)
            alpha = alpha0 + df * (alpha - alpha0) + p * ws
            beta = beta0 + df * (beta - beta0) + (1 - p) * wf
            t = max(min(t, 0.99), 0.01)
            logl += (
                loggamma(alpha + beta)
                - loggamma(alpha)
                - loggamma(beta)
                + (alpha - 1) * np.log(t)
                + (beta - 1) * np.log(1.0 - t)
            )

        return -logl

    @staticmethod
    def gradients(x, *args):
        """
        The gradient of the log-likelihood function
        """
        grads = np.zeros_like(x)
        trust_history, perf_history, _df = args
        num_sites = len(perf_history)
        diff_alpha = 0
        diff_beta = 0
        alpha0, beta0, ws, wf = x

        grads_alpha_i_ws = 0.0
        grads_beta_i_ws = 0.0

        for i in range(num_sites):
            df = _df.get_value(i)
            diff_alpha = df * diff_alpha + ws * perf_history[i]
            diff_beta = df * diff_beta + wf * (1 - perf_history[i])
            alpha_i = alpha0 + diff_alpha
            beta_i = beta0 + diff_beta

            grads_alpha_i_ws = df * grads_alpha_i_ws + perf_history[i]
            grads_beta_i_ws = df * grads_beta_i_ws + (1 - perf_history[i])

            if trust_history[i] < 0:
                # If a placeholder is encountered, do not compute gradients here, since feedback was not queried at this
                # site
                continue

            digamma_both = digamma(alpha_i + beta_i)
            digamma_alpha = digamma(alpha_i)
            digamma_beta = digamma(beta_i)
            tf = clamp(trust_history[i], _min=0.01, _max=0.99)
            log_fi = log(tf)
            log_minus_fi = log(1.0 - tf)

            grads[0] += digamma_both - digamma_alpha + log_fi
            grads[1] += digamma_both - digamma_beta + log_minus_fi
            grads[2] += (digamma_both - digamma_alpha + log_fi) * grads_alpha_i_ws
            grads[3] += (digamma_both - digamma_beta + log_minus_fi) * grads_beta_i_ws

        return -grads

    @staticmethod
    def get_initial_guess():
        return {"alpha0": 1.0, "beta0": 1.0, "ws": WS, "wf": WF}

    def estimate(
        self,
        trust: float,
        performance: float,
        trust_params: Dict,
        verbose: bool = False,
    ):
        """
        Estimates the trust params given the data
        :param trust: The trust feedback given after the currents search site is completed
        :param performance: The performance of the robot at the current site
        :param trust_params: The estimate at the last time step
        :param verbose: whether to print debugging data
        :return:
        """
        self.performance_history.append(performance)
        self.trust_feedback.append(trust)

        x0 = np.ones((4,), dtype=float)

        def fun(x):
            return self.neg_log_likelihood(
                x, self.trust_feedback, self.performance_history, self.df
            )

        def grad(x):
            return self.gradients(
                x, self.trust_feedback, self.performance_history, self.df
            )

        bnds = ((1, 200), (1, 200), (0.1, 200), (0.1, 200))
        # bnds = ((1, 200), (1, 200))

        res = minimize(fun, x0, jac=grad, method="SLSQP", bounds=bnds)
        optim_params = res.x
        return {
            "alpha0": optim_params[0],
            "beta0": optim_params[1],
            "ws": optim_params[2],
            "wf": optim_params[3],
        }


def clamp(x, _min, _max):
    return max(min(x, _max), _min)
