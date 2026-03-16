from typing import Dict

from classes.DiscountFactors import ConstantDF


class TrustEstimator:
    def __init__(self, discount_factor: ConstantDF, trust_params: Dict):
        self.df = discount_factor
        self.alpha_difference = 0
        self.beta_difference = 0
        self.trust_params = trust_params
        self.performance_history = []

    def clear(self):
        self.performance_history.clear()

    def update_params(self, trust_params: Dict):
        self.trust_params = trust_params

    def get_trust(self, performance: int, site_number: int):
        self.performance_history.append(performance)
        self.alpha_difference = 0
        self.beta_difference = 0
        for i in range(site_number + 1):
            self.alpha_difference = (
                self.df.get_value(i) * self.alpha_difference
                + self.trust_params["ws"] * self.performance_history[i]
            )

            self.beta_difference = self.df.get_value(
                i
            ) * self.beta_difference + self.trust_params["wf"] * (
                1 - self.performance_history[i]
            )

        alpha = self.trust_params["alpha0"] + self.alpha_difference
        beta = self.trust_params["beta0"] + self.beta_difference

        trust = alpha / (alpha + beta)

        return trust, alpha, beta
