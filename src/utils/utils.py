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
                            / np.sum(df['nat_tv_product_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))


            elif var == 'digital_2020_2022Q1_imp':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['digital_2020_2022Q1_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))

            elif var == 'nat_tv_wo2020_product_imp_sov':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['nat_tv_product_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))


            elif var == 'nat_tv_wo2020_vfm_imp_sov':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['nat_tv_vfm_spend'])
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


class Step4:
    def __init__(self, df):
        self.df = df.fillna(0)
        p_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        #self.df['date'] = pd.to_datetime(self.df['date'])
        #self.df = self.df.set_index('date')
        self.df['year'] = self.df.index.year
        self.df['quarter'] = self.df.index.quarter

        # TODO: убрать хардкод!

        last_roi = pd.read_excel('data/interim/step3/roi_11.xlsx')  # откуда берем roi?

        for file in [i for i in os.listdir('data/interim/step3/') if '0' in i and 'iteration' in i]:
            for p_level in p_levels:
                if file.split('_iteration_0_trans_step_3_res.xlsx')[0] in last_roi['chanel'].values:
                    df = pd.read_excel('data/interim/step3/' + file)

                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df.dropna(inplace=True)
                    df = df[df['coef'] > 0]
                    df = df[df['ban_roi'] == 0]
                    df = df[df['p_value'] <= p_level]
                    df = df[df['ban_adstock'] == 0]

                    if len(df) > 0:
                        df['weight'] = df['final_metric'] / df['final_metric'].sum()
                        df.to_excel('data/interim/step3/' + file, index=False)
                        break
                else:
                    continue

        return None

    def process_files(self, df2):

        df = df2.fillna(0)

        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        data = df.copy()

        latest_weights = pd.read_excel('data/interim/step3/' + "coefs_11.xlsx")

        last_roi = pd.read_excel('data/interim/step3/' + 'roi_11.xlsx')

        dates_list = sorted(list(map(lambda x: (x[0], [x[1]]), set([(year, quarter) \
                                                                    for year, quarter in
                                                                    zip(data['year'], data['quarter'])]))), \
                            key=lambda x: (x[0], x[1]))


        for file in [i for i in os.listdir('data/interim/step3/') if '11' in i and 'iteration' in i]:

            model_df = pd.read_excel('data/interim/step3/' + file)

            if file.split('_iteration_11_trans_step_3_res.xlsx')[0] in last_roi['chanel'].values:

                if 'weight' in model_df.columns:

                    model_df.replace([np.inf, -np.inf], np.nan, inplace=True)

                    model_df = model_df.reset_index().drop('index', axis=1)

                    groups = [
                                 (2020, range(1, 5)),
                                 (2021, range(1, 5)),
                                 (2022, range(1, 5))] + dates_list

                    current_var = file.split('iteration')[0][:-1]

                    for index, row in (model_df.iterrows()):

                        cumulative_df = pd.DataFrame(columns=df.columns)

                        for group in groups:

                            df = data.copy()

                            df = df.fillna(0)

                            df[f"{current_var}_c"] = Carryover(strength=row['strength'], length=int(row['length'])).fit(
                                np.array(df[current_var]).reshape(-1, 1)).transform(
                                np.array(df[current_var]).reshape(-1, 1))

                            df[f"{current_var}_trans"] = Saturation(x0=row['x0'], alpha=row['alpha']).fit(
                                np.array(df[f"{current_var}_c"]).reshape(-1, 1)).transform(
                                np.array(df[f"{current_var}_c"]).reshape(-1, 1))

                            temp_df = df[(df['year'] == group[0]) & (df['quarter'].isin(group[1]))]

                            # display(temp_df.index[0])

                            if current_var == 'gis_imp':

                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['gis_spend'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            elif current_var == 'reg_tv_imp':
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['reg_tv_spend'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            elif current_var == 'OOH_imp':
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['OOH_spend'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            elif current_var == 'full_yandex_maps_imp':
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['full_yandex_maps_spend'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            elif current_var == 'nat_tv_wo2020_angus_imp_norm_sov':
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['nat_tv_angus_spend'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            elif current_var == 'nat_tv_wo2020_product_imp_sov':
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['nat_tv_product_spend'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            elif current_var == 'nat_tv_wo2020_vfm_imp_sov':
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['nat_tv_vfm_spend'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            elif current_var == 'digital_none_youtube_imp':
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['digital_none_youtube_spend'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            elif current_var == 'digital_2020_2022Q1_imp':
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['digital_2020_2022Q1_spend'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            elif current_var == 'digital_imp_youtube':
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df['digital_spend_youtube'])
                                       }

                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            else:
                                roi = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef']) * np.mean(
                                    temp_df['avg_check']) \
                                                    / np.sum(temp_df[current_var])
                                       }
                                impact = {current_var: np.sum(temp_df[f"{current_var}_trans"] * row['coef'])

                                          }

                            roi = pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0)

                            impact = pd.DataFrame.from_dict(impact, orient='index').reset_index().fillna(0)

                            if group[0] == 2020 and isinstance(group[1], range):

                                roi.columns = ['chanel', 'roi_2020']
                                model_df.loc[index, 'roi_2020'] = roi['roi_2020'].values[0]
                                impact.columns = ['chanel', 'impact_2020']
                                model_df.loc[index, 'impact_2020'] = impact['impact_2020'].values[0]

                            elif group[0] == 2021 and isinstance(group[1], range):

                                roi.columns = ['chanel', 'roi_2021']
                                model_df.loc[index, 'roi_2021'] = roi['roi_2021'].values[0]
                                impact.columns = ['chanel', 'impact_2021']
                                model_df.loc[index, 'impact_2021'] = impact['impact_2021'].values[0]

                            elif group[0] == 2022 and isinstance(group[1], range):

                                roi.columns = ['chanel', 'roi_2022']
                                model_df.loc[index, 'roi_2022'] = roi['roi_2022'].values[0]
                                impact.columns = ['chanel', 'impact_2022']
                                model_df.loc[index, 'impact_2022'] = impact['impact_2022'].values[0]

                            else:

                                roi.columns = ['chanel', f'roi_{str(group[0])}_Q{str(group[1][0])}']

                                model_df.loc[index, f'roi_{str(group[0])}_Q{str(group[1][0])}'] \
                                    = roi[f'roi_{str(group[0])}_Q{str(group[1][0])}'].values[0]

                                impact.columns = ['chanel', 'impact_' + f'roi_{str(group[0])}_Q{str(group[1][0])}']

                                model_df.loc[index, 'impact_' + f'roi_{str(group[0])}_Q{str(group[1][0])}'] = \
                                impact['impact_' + f'roi_{str(group[0])}_Q{str(group[1][0])}'].values[0]

                        del temp_df

                    model_df['weighted_strength'] = np.round(np.sum(model_df['weight'] * model_df['strength']), 2)

                    model_df['weighted_length'] = int(np.round(np.sum(model_df['weight'] * model_df['length'])))

                    model_df['weighted_roi_2020'] = np.round(np.sum(model_df['weight'] * model_df['roi_2020']), 2)

                    model_df['weighted_roi_2021'] = np.round(np.sum(model_df['weight'] * model_df['roi_2021']), 2)

                    model_df['weighted_roi_2022'] = np.round(np.sum(model_df['weight'] * model_df['roi_2022']), 2)

                    model_df['weighted_' + f'roi_{str(group[0])}_Q + {str(group[1][0])}'] = np.round(
                        np.sum(model_df['weight'] * \
                               model_df[f'roi_{str(group[0])}_Q{str(group[1][0])}']), 2)

                    model_df['weighted_impact_2020'] = np.round(np.sum(model_df['weight'] * model_df['impact_2020']), 2)

                    model_df['weighted_impact_2021'] = np.round(np.sum(model_df['weight'] * model_df['impact_2021']), 2)

                    model_df['weighted_impact_2022'] = np.round(np.sum(model_df['weight'] * model_df['impact_2022']), 2)

                    model_df['weighted_impact_' + f'roi_{str(group[0])}_Q + {str(group[1][0])}'] = np.round(
                        np.sum(model_df['weight'] * \
                               model_df['impact_' + f'roi_{str(group[0])}_Q{str(group[1][0])}']), 2)

                    model_df.to_excel('data/interim/step3/'+current_var + '_w_ROI_w_Impact_all' + '.xlsx', index=False)

            else:

                continue

        return self