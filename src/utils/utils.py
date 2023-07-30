"""
Реализация основных классов и функций
"""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import convolve2d
import statsmodels.formula.api as sm
from matplotlib import pyplot as plt
import os
pallete = plt.get_cmap('Set2')
import warnings
warnings.filterwarnings('ignore')


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


class Step2:
    def __init__(self, current_region, df):
        self.current_region = current_region
        self.df = df.fillna(0)

        self.metric_files = [i for i in os.listdir('data/interim') if
                             i not in ('.gitkeep') and '.xlsx' in i]
        self.matrix = None

        self.context_vars = ['stores', 'seasonality', 'competitors_list_tv', 'new_covid',
                             'sales_qsr',
                             'dish_qnt_reg_negative',
                             'average_price_dish_region_smooth_5', 'price_lag_new_smooth_40',
                             'dummy_apr']

        self.paid_vars_imp = ["gis_imp",
                              'final_ooh',
                              "final_tm",
                              "reg_tv_imp",
                              "full_yandex_maps_imp",
                              "OOH_imp",
                              "final_posm",
                              'digital_2020_2022Q1_imp',
                              "nat_tv_wo2020_product_imp_sov",
                              "nat_tv_wo2020_vfm_imp_sov",
                              "final_ap",
                              'digital_none_youtube_imp']

        for i in self.paid_vars_imp:
            if np.sum(self.df[i]) == 0:
                print(i, 'is dropped!')
                self.df = self.df.drop(i, axis=1)
                self.paid_vars_imp.remove(i)

    def process_files(self, top_n=1):
        for i, file in enumerate(self.metric_files):
            print('-------------------------------------------')
            print(f"file {file} in progress, file number {i + 1}")
            print('-------------------------------------------')

            metric_df = pd.read_excel('data/interim/'+file)
            strength = list(metric_df['strength'].values)[:top_n]
            length = list(metric_df['length'].values)[:top_n]
            x0 = list(metric_df['x0'].values)[:top_n]
            alpha = list(metric_df['alpha'].values)[:top_n]

            # For the first file, initialize matrix. For other files, concatenate to existing matrix
            if i == 0:
                self.matrix = pd.DataFrame(strength, index=['strength'], columns=[file[:-15]]) \
                    .append(pd.DataFrame(length, index=['length'], columns=[file[:-15]])) \
                    .append(pd.DataFrame(x0, index=['x0'], columns=[file[:-15]])) \
                    .append(pd.DataFrame(alpha, index=['alpha'], columns=[file[:-15]]))
            else:
                self.matrix = pd.concat((self.matrix,
                                         pd.DataFrame(strength, index=['strength'], columns=[file[:-15]]) \
                                         .append(pd.DataFrame(length, index=['length'], columns=[file[:-15]])) \
                                         .append(pd.DataFrame(x0, index=['x0'], columns=[file[:-15]])) \
                                         .append(pd.DataFrame(alpha, index=['alpha'], columns=[file[:-15]]))),
                                        axis=1)
        self.matrix = self.matrix.reset_index().rename(columns={'index': 'parameter'})
        self.matrix.to_excel("data/processed/matrix_params.xlsx", index=False)
        return self

    def fit(self):

        # self.df = pd.read_excel('final_df.xlsx').fillna(0)
        #self.df = df
        self.trans_var_list = self.matrix.columns[1:]

        for i in range(len(self.trans_var_list)):
            marketing_var = self.trans_var_list[i]
            self.df[f"{marketing_var}_c"] = Carryover(strength=self.matrix[marketing_var].values[0],
                                                      length=int(self.matrix[marketing_var].values[1])).fit(
                np.array(self.df[marketing_var]).reshape(-1, 1)).transform(
                np.array(self.df[marketing_var]).reshape(-1, 1))

            self.df[f"{marketing_var}_trans"] = Saturation(x0=self.matrix[marketing_var].values[2],
                                                           alpha=self.matrix[marketing_var].values[3]).fit(
                np.array(self.df[f"{marketing_var}_c"]).reshape(-1, 1)).transform(
                np.array(self.df[f"{marketing_var}_c"]).reshape(-1, 1))

        self.model = sm.ols(formula=formula(set(list([i + '_trans' for i in self.trans_var_list]) + self.context_vars)),
                            data=self.df).fit(method="pinv")

        print(self.model.summary())

        self.df.to_excel('data/processed/df_modeling.xlsx', index=False)

        return self

    def ROI(self):

        coef = pd.DataFrame(self.model.params).T
        print(coef)
        ROI = pd.DataFrame([[0, 0]], columns=['index', 0])

        df = pd.read_excel('data/processed/df_modeling.xlsx')

        for var in self.matrix.columns[1:].to_list():

            if var == 'gis_imp':

                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['gis_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))


            elif var == 'reg_tv_imp':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['reg_tv_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))


            elif var == 'OOH_imp':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['OOH_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))


            elif var == 'full_yandex_maps_imp':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['full_yandex_maps_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))


            elif var == 'nat_tv_wo2020_imp_sov':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['nat_tv_spend_2021_2022'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))


            elif var == 'digital_2020_2022Q1_imp':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['digital_2020_2022Q1_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))


            elif var == 'digital_none_youtube_imp':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['digital_none_youtube_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))


            else:
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df[var])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))

        ROI.columns = ['chanel', 'roi']
        ROI = ROI[1:]
        ROI.to_excel('data/processed/ROI_Step_2.xlsx')
        print(ROI.round(0))

        return self

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
