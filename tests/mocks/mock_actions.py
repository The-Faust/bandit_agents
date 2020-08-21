from scipy.stats import gamma


class GammaGiver:
    def __init__(self, alpha, loc):
        self.alpha = alpha
        self.loc = loc

    def do(self):
        return gamma.rvs(self.alpha, self.loc)


mockActions = {
    'sell_a': GammaGiver(4, 10).do,
    'sell_b': GammaGiver(4, 10).do,
    'sell_c': GammaGiver(4, 30).do,
    'sell_d': GammaGiver(4, 10).do,
    'sell_e': GammaGiver(4, 10).do,
    'sell_f': GammaGiver(4, 10).do,
    'sell_g': GammaGiver(4, 10).do,
    'sell_h': GammaGiver(4, 10).do,
    'sell_i': GammaGiver(4, 40).do,
    'sell_j': GammaGiver(4, 10).do
}