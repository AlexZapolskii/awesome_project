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

# Загрузка данных:

if __name__ == '__main__'
    PATH = "data/raw/total_df.csv"
    df = load_transform_dataset(PATH)

    # TODO: call optimizer
    #       В оптимизаторе перебираются параметры
    #       Сохраняется резульатат - ексели (в interim или processed)




# Параметры для оптимизации