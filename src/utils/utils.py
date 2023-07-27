"""
Реализация основных классов и функций
"""


# class carryover (cumulative effect of advertising)
class Carryover(BaseEstimator, TransformerMixin):
    def __init__(self, strength=0.9, length=3):
        self.strength = strength
        self.length = length

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window_ = (self.strength ** np.arange(self.length + 1)).reshape(
            -1, 1
        )
        return self

    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window_)
        if self.length > 0:
            convolution = convolution[: -self.length]
        return convolution


class Saturation(BaseEstimator, TransformerMixin):
    def __init__(self, x0=10000, alpha=0.000002):
        self.alpha = alpha
        self.x0 = x0

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)  # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)  # from BaseEstimator
        return (1 / (1 + np.exp(-self.alpha * (X - self.x0)))) - (
            1 / (1 + np.exp(-self.alpha * (0 - self.x0)))
        )


def formula(var_list):
    col_str = ""
    for var in var_list:
        col_str = str(var) + "+" + col_str
    col_str = col_str[:-1]
    col_str = "sales~" + col_str
    return col_str


def smoothing(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result
