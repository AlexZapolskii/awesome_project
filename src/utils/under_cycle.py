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
from utils.utils import *
from utils.enums import *
from src.utils.load_data import  load_transform_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from tqdm.notebook import tqdm
import itertools
from PIL import Image, ImageDraw, ImageFont
import glob
import os
import shutil

# Задаем сетки для подбора параметров

subnumbers = 5
a_x = np.linspace(20, 50, 4)
b_x = np.linspace(70, 99, 4)

percentiles_border_combs = []  # x

for r in itertools.product(a_x, b_x):
    if np.all(np.diff(r) >= 48):
        percentiles_border_combs.append(r)

a_y = np.linspace(0.01, 0.51, 4)
b_y = np.linspace(0.49, 0.99, 4)

combs = []  # y
for r in itertools.product(a_y, b_y):
    if np.all(np.diff(r) <= 0.9) and np.all(np.diff(r) > 0) and np.all(np.diff(r) >= 0.45):
        combs.append(r)

strength = np.linspace(0, 0.8, subnumbers)
length = [2, 4, 6, 8, 10, 12]

var = 'final_tm'
s = 0.4
l = 6
i = (0.17666666666666667, 0.6566666666666666)
percentiles_border = (20.0, 79.66666666666667)

ans = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      columns = ['coef', 'percentile_values', 'percentiles', 'mape', 'r2', 'r2_adj', 'p_value', 'negative_signs', 'wrong_roi', 'wrong_adstock', f"{var}_roi", 'percentile_1', 'percentile_2', 'max_value_adstock', 'ban_roi', 'ban_adstock'])

df = load_transform_dataset('../data/raw/total_df.csv')

# Вот здесь идет какое то усреднение скользящим окном для переменной competitors_list_tv
# TODO: - как сделать, чтобы не перезатиралось поле (и неперезагружать датасет заново)

# Видим, что для переменной competitors_list_tv параметры фиксированны
df['competitors_list_tv'] = Carryover(strength=0.8, length=12) \
    .fit(np.array(df['competitors_list_tv']) \
         .reshape(-1, 1)) \
    .transform(np.array(df['competitors_list_tv']) \
               .reshape(-1, 1))


# Здесь уже Carryover параметризуется
# Параметры: var (переменная), strength = s, length = l (параметры Carryover)
df[f"{var}_c"]  = Carryover(strength = s, length = l).fit(np.array(df[var]).reshape(-1,1)).transform(np.array(df[var]).reshape(-1,1))
index = range(len(df[f'{var}_c'][df[f'{var}_c'] > 0].sort_values()))
percentiles = [int(np.percentile(np.array(index), i)) for i in percentiles_border]   # не конфликтует с i в цикле?
x_data = [df[f'{var}_c'][df[f'{var}_c']>0].sort_values()[p] for p in percentiles]
#x_data = [np.max(df[f'{var}_c'][df[f'{var}_c']>0].sort_values()) * p for p in combs[0]]
max_value_adstock = np.max(df[f'{var}_c'][df[f'{var}_c']>0].sort_values())

# i = (0.17666666666666667, 0.6566666666666666)  не используется combs

# Next Block

y_data = i

data = pd.DataFrame([x_data, y_data]).T
data.columns = ['x', 'y']

data.iloc[0, 1] = data.iloc[0, 1] + 1e-10
data.iloc[1, 1] = data.iloc[1, 1] + 1e-11
# data.iloc[2, 1] = data.iloc[2, 1] + 1e-12

data['y'] = data['y'] + 1e-3
data['z'] = np.log(1 / data['y'] - 1)

x0 = (data['z'][0] * data['x'][1] - data['z'][1] * data['x'][0]) / (data['z'][0] - data['z'][1])

if x0 == np.inf or x0 == np.nan or x0 == 0:   # обработка пропусков или inf
    x0 = x0 + 1e-10

alpha_1 = np.round(data['z'][0] / (x0-data['x'][0]), 15)
alpha_2 = data['z'][1] / (x0-data['x'][1])

alpha = alpha_1