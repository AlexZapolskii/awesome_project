from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from scipy.signal import convolve2d
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import re
import pandas as pd
import os
import statsmodels.formula.api as sm
from scipy.ndimage.interpolation import shift
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
pallete = plt.get_cmap('Set2')
import warnings
from datetime import date
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from tqdm.notebook import tqdm
import itertools
from PIL import Image, ImageDraw, ImageFont
import glob
import os
import shutil

from utils.enums import (competitors_list_w_o_tv, competitors_list_tv, digital_list, digital_spend_list,
                         paid_vars_imp, paid_vars_spend, context_vars, TV_Digital_OOH_Geo)


# class carryover (cumulative effect of advertising)
class Carryover:
    def __init__(self, strength=0.9, length=3):
        self.strength = strength
        self.length = length

    def fit(self, X):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window_ = (self.strength ** np.arange(self.length + 1)).reshape(-1, 1)
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

    def fit(self, X):
        X = check_array(X)
        self._check_n_features(X, reset=True)  # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)  # from BaseEstimator
        return (1 / (1 + np.exp(-self.alpha * (X - self.x0)))) - (1 / (1 + np.exp(-self.alpha * (0 - self.x0))))


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

if __name__ == '__main__':
    CURRENT_REGION = 'юг'
    FILE_LOCATION = "../data/raw/total_df.csv"
    df = pd.read_csv(FILE_LOCATION).fillna(0)

    df = df[df['region'] == CURRENT_REGION]
    df['final_posm'] = df['final_posm'] + df['final_booking_promo']

    df["competitors_list_tv"] = df[competitors_list_tv].sum(axis=1)
    df["competitors_list_w_o_tv"] = df[competitors_list_w_o_tv].sum(axis=1)

    paid_vars_spend_WO_ap_tm = [i for i in paid_vars_imp if 'final_ap' not in i and 'final_tm' not in i]

    df['mc_leave'] = (df.date >= '2022-03-14').astype('int')
    df['lockdown'] = ((df.date >= "2021-10-28") & (df.date <= "2021-11-07")).astype('int')
    df['nat_tv_spend_2021_2022'] = np.where(pd.to_datetime(df['date']).dt.year >= 2021, df['nat_tv_spend'], 0)

    df = df[df.date >= "2020-01-01"]

    decomposition = seasonal_decompose(df[["sales"]], model="multiplicative", period=int(len(df) / 2))

    df["trend"] = range(len(df))
    df['seasonality'] = df['Seas_calc_2016_2022']
    df['nat_tv_wo2020_product_imp_sov'] = df['nat_tv_wo2020_product_imp_norm'] * 0.5 + df['SOV_product_norm'] * 0.5
    df['nat_tv_wo2020_vfm_imp_sov'] = df['nat_tv_wo2020_vfm_imp_norm'] * 0.5 + df['SOV_vfm_norm'] * 0.5

    df["digital_2020_2022Q1_imp"] = df[digital_list].sum(axis=1)
    df["digital_2020_2022Q1_imp"] = df[df.date <= '2022-04-01']['digital_2020_2022Q1_imp']

    df["digital_2020_2022Q1_spend"] = df[digital_spend_list].sum(axis=1)
    df["digital_2020_2022Q1_spend"] = df[df.date <= '2022-04-01']['digital_2020_2022Q1_spend']

    df["digital_none_youtube_imp"] = df[digital_list].sum(axis=1)
    df["digital_none_youtube_imp"] = df[df.date > '2022-04-01']['digital_none_youtube_imp']

    df["digital_none_youtube_spend"] = df[digital_spend_list].sum(axis=1)
    df["digital_none_youtube_spend"] = df[df.date > '2022-04-01']['digital_none_youtube_spend']

    df['avg_check'] = 1

    for i in paid_vars_imp:
        if df[i].sum() == 0:
            df = df.drop(i, axis=1)
            paid_vars_imp.remove(i)

    df.to_excel('../data/interim/final_df.xlsx')

