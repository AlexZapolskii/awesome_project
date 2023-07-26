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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from tqdm.notebook import tqdm
import itertools
from PIL import Image, ImageDraw, ImageFont
import glob
import os
import shutil

current_region = 'юг'

file = "../data/raw/total_df.xlsx"
df = pd.read_excel(file).fillna(0)
df.to_csv('total_df.csv', index = False)
file = "total_df.csv"
df = pd.read_csv(file).fillna(0)

df = df[df['region'] == current_region]
df['final_posm'] = df['final_posm'] + df['final_booking_promo']

df['date'] = pd.to_datetime(df['date'])
df['mc_leave'] = np.where(df['date'] >= '2022-03-14', 1, 0)
df['lockdown'] = np.where((df['date'] >= "2021-10-28") & (df['date'] <= "2021-11-07"), 1, 0)
df['nat_tv_spend_2021_2022'] = np.where(df['date'].dt.year >= 2021, df['nat_tv_spend'], 0)

df = df.set_index("date")
df = df.loc[df.index >= "2020-01-01"]
df = df[df['region'] == current_region]

decomposition = seasonal_decompose(df[["sales"]], model="multiplicative", period=int(len(df) / 2))
df["trend"] = range(len(df))

df['seasonality'] = df['Seas_calc_2016_2022']
df['nat_tv_wo2020_product_imp_sov'] = df['nat_tv_wo2020_product_imp_norm'] * 0.5 + df['SOV_product_norm'] * 0.5
df['nat_tv_wo2020_vfm_imp_sov'] = df['nat_tv_wo2020_vfm_imp_norm'] * 0.5 + df['SOV_vfm_norm'] * 0.5
df["digital_2020_2022Q1_imp"] = df[digital_list].sum(axis=1)
df["digital_2020_2022Q1_imp"] = df[df.index <= '2022-04-01']['digital_2020_2022Q1_imp']
df["digital_2020_2022Q1_spend"] = df[digital_spend_list].sum(axis=1)
df["digital_2020_2022Q1_spend"] = df[df.index <= '2022-04-01']['digital_2020_2022Q1_spend']
df["digital_none_youtube_imp"] = df[digital_list].sum(axis=1)
df["digital_none_youtube_imp"] = df[df.index > '2022-04-01']['digital_none_youtube_imp']
df["digital_none_youtube_spend"] = df[digital_spend_list].sum(axis=1)
df["digital_none_youtube_spend"] = df[df.index > '2022-04-01']['digital_none_youtube_spend']
df['avg_check'] = 1

for i in paid_vars_imp:
    if np.sum(df[i]) == 0:
        print(i)
        df = df.drop(i, axis = 1)
        paid_vars_imp.remove(i)

df.to_excel('final_df.xlsx')

