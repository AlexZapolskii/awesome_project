"""
Модуль для прогонки третьего шага
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
pallete = plt.get_cmap('Set2')
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_percentage_error
import itertools
from PIL import Image, ImageDraw
import os
from utils.utils import *
from utils.enums import *

def optimize_step3(df_source: pd.DataFrame, paid_vars_imp):
    """
    Третий ноутбучек
    """

    # n = 12   # задается кол-во итераций!
    n = 12      # оставили две для быстрого прогона

    minus_variables = []
    minus_variables_dict = {i: [] for i in range(n)}

    subnumbers = 5

    matrix = pd.read_excel('data/processed/matrix_params.xlsx').iloc[:,1: ]

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

    strength_arr = np.linspace(0, 0.8, subnumbers)

    length_arr = [2, 4, 6, 8, 10, 12]

    for number in (range(n)):    # итерации

        for var in (paid_vars_imp):   # переменные

            trans_var_list = [i for i in matrix.columns if i != var]

            ans = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], \
                               columns=['coef', 'percentile_values', 'percentiles', 'mape', 'r2', 'r2_adj', 'p_value',
                                        'negative_signs', 'wrong_roi', 'wrong_adstock', f"{var}_roi", 'percentile_1',
                                        'percentile_2', 'max_value_adstock', 'ban_roi', 'ban_adstock'])

            for s in (strength_arr):
                for l in (length_arr):
                    for comb in (combs):
                        for percentiles_border in percentiles_border_combs:

                            df = df_source.copy().fillna(0)

                            df['competitors_list_tv'] = Carryover(strength=0.8, length=12) \
                                .fit(np.array(df['competitors_list_tv']) \
                                     .reshape(-1, 1)) \
                                .transform(np.array(df['competitors_list_tv']) \
                                           .reshape(-1, 1))

                            df['final_posm'] = df['final_posm'] + df['final_booking_promo']

                            df["competitors_list_w_o_tv"] = df[competitors_list_w_o_tv].sum(axis=1)

                            context_vars = ['stores', 'seasonality', 'competitors_list_tv', 'new_covid', 'lockdown',
                                            'sales_qsr', 'test_mac_off', 'covid_dummy', 'back_to_school', 'big_hit',
                                            'McD_leave', 'ViT_2', 'dish_qnt_reg_negative',
                                            'average_price_dish_region_smooth_5', 'price_lag_new_smooth_40',
                                            'dummy_apr', 'comps_SOM']

                            df[f"{var}_c"] = Carryover(strength=s, length=l).fit(
                                np.array(df[var]).reshape(-1, 1)).transform(np.array(df[var]).reshape(-1, 1))
                            x_data = [np.max(df[f'{var}_c']) * (p / 100) for p in percentiles_border]

                            max_value_adstock = np.max(df[f'{var}_c'][df[f'{var}_c'] > 0].sort_values())

                            y_data = comb

                            data = pd.DataFrame([x_data, y_data]).T
                            data.columns = ['x', 'y']

                            data.iloc[0, 1] = data.iloc[0, 1] + 1e-10
                            data.iloc[1, 1] = data.iloc[1, 1] + 1e-11
                            # data.iloc[2, 1] = data.iloc[2, 1] + 1e-12

                            data['z'] = np.log(1 / data['y'] - 1)

                            x0 = (data['z'][0] * data['x'][1] - data['z'][1] * data['x'][0]) / (data['z'][0] - data['z'][1])

                            if x0 == np.inf or x0 == np.nan or x0 == 0:
                                x0 = x0 + 1e-10

                            alpha_1 = np.round(data['z'][0] / (x0 - data['x'][0]), 15)
                            alpha_2 = data['z'][1] / (x0 - data['x'][1])

                            alpha = alpha_1

                            df[f"{var}_trans"] = Saturation(x0, alpha).fit(
                                np.array(df[f"{var}_c"]).reshape(-1, 1)).transform(np.array(df[f"{var}_c"]).reshape(-1, 1))

                            for i in range(len(trans_var_list)):
                                marketing_var = trans_var_list[i]

                                df[f"{marketing_var}_c"] = Carryover(strength=matrix[marketing_var].values[0],
                                                                     length=int(matrix[marketing_var].values[1])).fit(
                                    np.array(df[marketing_var]).reshape(-1, 1)).transform(
                                    np.array(df[marketing_var]).reshape(-1, 1))

                                df[f"{marketing_var}_trans"] = Saturation(x0=matrix[marketing_var].values[2],
                                                                          alpha=matrix[marketing_var].values[3]).fit(
                                    np.array(df[f"{marketing_var}_c"]).reshape(-1, 1)).transform(
                                    np.array(df[f"{marketing_var}_c"]).reshape(-1, 1))

                            model = sm.ols(
                                formula=formula([f"{var}_trans"] + context_vars + [i + '_trans' for i in trans_var_list]),
                                data=df).fit(method="pinv")

                            paid_vars_spend_WO_ap_tm = [i for i in paid_vars_imp if
                                                        'final_ap' not in i and 'final_tm' not in i]

                            coef = pd.DataFrame(model.params).T

                            if var == 'gis_imp':
                                roi = {
                                    var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                                         / np.sum(df['gis_spend'])
                                    }

                            elif var == 'reg_tv_imp':
                                roi = {
                                    var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                                         / np.sum(df['reg_tv_spend'])
                                    }

                            elif var == 'OOH_imp':
                                roi = {
                                    var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                                         / np.sum(df['OOH_spend'])
                                    }

                            elif var == 'full_yandex_maps_imp':
                                roi = {
                                    var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                                         / np.sum(df['full_yandex_maps_spend'])
                                    }

                            elif var == 'nat_tv_wo2020_product_imp_sov':
                                roi = {
                                    var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                                         / np.sum(df['nat_tv_product_spend'])
                                    }

                            elif var == 'nat_tv_wo2020_vfm_imp_sov':
                                roi = {
                                    var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                                         / np.sum(df['nat_tv_vfm_spend'])
                                    }

                            elif var == 'digital_none_youtube_imp':
                                roi = {
                                    var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                                         / np.sum(df['digital_none_youtube_spend'])
                                    }

                            elif var == 'digital_2020_2022Q1_imp':
                                roi = {
                                    var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                                         / np.sum(df['digital_2020_2022Q1_spend'])
                                    }

                            else:
                                roi = {
                                    var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                                         / np.sum(df[var])
                                    }

                            roi = pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0)

                            roi.columns = ['chanel', 'roi']

                            wrong_roi = np.where(0.5 < roi['roi'].values < 5, 1,
                                                 np.where(roi['roi'].values > 5, roi['roi'].values / 5,
                                                          np.where(roi['roi'].values < 0.5, 0.5 / roi['roi'].values, 0)))[0]

                            ban_roi = np.where(roi['roi'].values > 10, 1, 0)[0]

                            wrong_adstock = np.where(s > 0.95, np.round(s / 0.95, 5), 1)

                            y_pred = model.predict(
                                df[[f"{var}_trans"] + context_vars + [i + '_trans' for i in trans_var_list]])
                            y_pred = y_pred.reset_index()[0]
                            y_pred = y_pred.fillna(np.mean(y_pred))

                            mape = mean_absolute_percentage_error(df["sales"], y_pred)
                            r2 = np.round(model.rsquared, 5)
                            r2_adj = np.round(model.rsquared_adj, 5)
                            p_values_df = pd.DataFrame(model.pvalues).reset_index().rename(
                                columns={'index': 'variables', 0: 'p'})
                            p_value = np.round(p_values_df[p_values_df['variables'] == f"{var}_trans"]['p'].values[0], 15)

                            max_value_adstock = np.max(df[f'{var}_c'][df[f'{var}_c'] > 0].sort_values())

                            linear_coef = coef[f"{var}_trans"].values[0]

                            if var == 'gis_imp':
                                ban_adstock = not (0 <= s <= 0.3)

                            elif var == 'final_ooh':
                                ban_adstock = not (0 <= s <= 0.85)

                            elif var == 'final_tm':
                                ban_adstock = not (0 <= s <= 0.85)

                            elif var == 'reg_tv_imp':
                                ban_adstock = not (0.15 <= s <= 0.85)

                            elif var == 'full_yandex_maps_imp':
                                ban_adstock = not (0 <= s <= 0.3)

                            elif var == 'OOH_imp':
                                ban_adstock = not (0 <= s <= 0.7)

                            elif var == 'final_posm':
                                ban_adstock = not (0 <= s <= 0.85)

                            elif var == 'digital_2020_2022Q1_imp':
                                ban_adstock = not (0.15 <= s <= 0.85)

                            elif var == 'nat_tv_wo2020_product_imp_sov':
                                ban_adstock = not (0.15 <= s <= 0.85)

                            elif var == 'nat_tv_wo2020_vfm_imp_sov':
                                ban_adstock = not (0.15 <= s <= 0.85)

                            elif var == 'final_ap':
                                ban_adstock = not (0 <= s <= 0.3)

                            elif var == 'final_booking_promo':
                                ban_adstock = not (0 <= s <= 0.7)

                            elif var == 'digital_none_youtube_imp':
                                ban_adstock = not (0.15 <= s <= 0.85)

                            ans = ans.append(pd.DataFrame([[linear_coef, x_data, y_data, mape, r2, r2_adj, p_value,
                                                            f'model_{s}_{l}_{np.round(x0, 5)}_{alpha}',
                                                            np.sum(coef[f"{var}_trans"] < 0), wrong_roi, wrong_adstock,
                                                            np.round(roi['roi'].values[0], 2), percentiles_border[0],
                                                            percentiles_border[1], max_value_adstock, ban_roi,
                                                            ban_adstock]],
                                                          columns=['coef', 'percentile_values', 'percentiles', 'mape', 'r2',
                                                                   'r2_adj', 'p_value', 'model', 'negative_signs',
                                                                   'wrong_roi', 'wrong_adstock',
                                                                   roi['chanel'].values[0] + '_roi', 'percentile_1',
                                                                   'percentile_2', 'max_value_adstock', 'ban_roi',
                                                                   'ban_adstock']))
                            # конец цикла, df - удаляется
                            # ans - получили

            context_vars = []

            ans = ans[1:].fillna(0)

            ans['mape_norm'] = ans['mape'] / np.mean(ans['mape'])
            ans['r2_adj_norm'] = ans['r2_adj'] / np.mean(ans['r2_adj'])
            ans['p_value'] = ans['p_value']  # / np.mean(ans['p_value'])

            ans['p_value_norm'] = (
            (ans['p_value'] / (np.mean(ans['p_value']) + 1e-10)))  # ans['p_value'] / np.mean(ans['p_value'])

            ans['final_metric'] = ans['r2_adj_norm'] / ans['p_value_norm'] / ans['mape_norm'] / ans['wrong_roi'] / ans[
                'wrong_adstock']
            ans = ans.sort_values('final_metric', ascending=False)

            ans['strength'] = pd.to_numeric(ans['model'].apply(lambda x: x.split('_')[1:][0]))
            ans['length'] = pd.to_numeric(ans['model'].apply(lambda x: x.split('_')[1:][1]))
            ans['x0'] = pd.to_numeric(ans['model'].apply(lambda x: x.split('_')[1:][2]))
            ans['alpha'] = ans['model'].apply(lambda x: x.split('_')[1:][3])
            ans['percentile_y_1'] = ans['percentiles'].apply(lambda x: np.round(x[0], 5))
            ans['percentile_y_2'] = ans['percentiles'].apply(lambda x: np.round(x[1], 5))
            if number <= 6:
                ans.to_excel(f'{f"data/interim/step3/{var}_trans_step_3"}_res.xlsx', index=False)
                ans.to_excel(f'{f"data/interim/step3/{var}_iteration_{number}_trans_step_3"}_res.xlsx', index=False)

                # вот тут генерится результат по переменной в эксель
                # {var}_trans_step_3"
                # var}_iteration_{number}_trans_step_3 еще итерация добавляется

            elif number >= 6:

                if len(ans[ans['negative_signs'] == 0]) == 0:

                    minus_variables.append(var)

                    ans.to_excel(f'{f"data/interim/step3/{var}_trans_step_3"}_res.xlsx', index=False)
                    ans.to_excel(f'{f"data/interim/step3/{var}_iteration_{number}_trans_step_3"}_res.xlsx', index=False)

                    # еще результат
                else:

                    ans.to_excel(f'{f"data/interim/step3/{var}_trans_step_3"}_res.xlsx', index=False)
                    ans.to_excel(f'{f"data/interim/step3/{var}_iteration_{number}_trans_step_3"}_res.xlsx', index=False)

                    # и еще запись
            #break    # TODO: убрать break! -все должно отрабатывать

        top_n = 1

        files_res = [i for i in os.listdir('data/interim/step3') if f"iteration_{number}_" in i]

        if number < 6:

            metric_df = pd.read_excel('data/interim/step3/'+ files_res[0])   # чтение откуда?
            metric_df = metric_df[metric_df['ban_roi'] == 0]
            metric_df = metric_df[metric_df['ban_adstock'] == 0]
            metric_df = metric_df[metric_df['negative_signs'] == 0]


        elif number >= 6:

            metric_df = pd.read_excel('data/interim/step3/'+ files_res[0])
            metric_df = metric_df[metric_df['ban_adstock'] == 0]
            metric_df = metric_df[metric_df['ban_roi'] == 0]

        strength = list(metric_df['strength'].values)[:top_n]
        length = list(metric_df['length'].values)[:top_n]
        x0 = list(metric_df['x0'].values)[:top_n]
        alpha = list(metric_df['alpha'].values)[:top_n]

        matrix = pd.DataFrame(strength, index=['strength'], columns=['test']) \
            .append(pd.DataFrame(length, index=['length'], columns=['test'])) \
            .append(pd.DataFrame(x0, index=['x0'], columns=['test'])) \
            .append(pd.DataFrame(alpha, index=['alpha'], columns=['test']))

        files_n = 1

        wrong_roi = []

        for file in (files_res):
            print('-------------------------------------------')
            print(f"file {file} in progress, file number {files_n}")
            print('-------------------------------------------')
            if file in os.listdir('data/interim/step3'):

                metric_df = pd.read_excel('data/interim/step3/'+ file)

                metric_df = metric_df[metric_df['ban_roi'] == 0]
                metric_df = metric_df[metric_df['ban_adstock'] == 0]

            else:
                continue

            strength = list(metric_df['strength'].values)[:top_n]
            length = list(metric_df['length'].values)[:top_n]
            x0 = list(metric_df['x0'].values)[:top_n]
            alpha = list(metric_df['alpha'].values)[:top_n]
            matrix = pd.concat(
                (matrix, pd.DataFrame(strength, index=['strength'], columns=[file[:-22 - len("_iteration_0")]]) \
                 .append(pd.DataFrame(length, index=['length'], columns=[file[:-22 - len("_iteration_0")]])) \
                 .append(pd.DataFrame(x0, index=['x0'], columns=[file[:-22 - len("_iteration_0")]])) \
                 .append(pd.DataFrame(alpha, index=['alpha'], columns=[file[:-22 - len("_iteration_0")]]))), axis=1)

            files_n += 1

        matrix = matrix.drop('test', axis=1).reset_index().rename(columns={'index': 'parameter'})
        matrix = matrix.dropna(axis=1, how='all')
        # display(matrix)

        # куда записывать matrix_params?
        matrix.to_excel(f"data/interim/step3/matrix_params_{number}.xlsx", index=False)

        matrix = pd.read_excel(f"data/interim/step3/matrix_params_{number}.xlsx").iloc[:, 1:]

        df = df_source.copy().fillna(0)

        df['competitors_list_tv'] = Carryover(strength=0.8, length=12) \
            .fit(np.array(df['competitors_list_tv']) \
                 .reshape(-1, 1)) \
            .transform(np.array(df['competitors_list_tv']) \
                       .reshape(-1, 1))

        df["competitors_list_w_o_tv"] = df[competitors_list_w_o_tv].sum(axis=1)

        context_vars = ['stores', 'seasonality', 'competitors_list_tv', 'new_covid', 'lockdown',
                        'sales_qsr', 'test_mac_off', 'covid_dummy', 'back_to_school', 'big_hit',
                        'McD_leave', 'ViT_2', 'dish_qnt_reg_negative',
                        'average_price_dish_region_smooth_5', 'price_lag_new_smooth_40',
                        'dummy_apr', 'comps_SOM']

        m_columns = [i[:-1] if i[-1] == '_' else i for i in matrix.columns]

        matrix.columns = m_columns

        for i in matrix.columns:
            var = i

            df[f"{var}_c"] = Carryover(strength=matrix[var].values[0], length=int(matrix[var].values[1])).fit(
                np.array(df[var]).reshape(-1, 1)).transform(np.array(df[var]).reshape(-1, 1))

            df[f"{var}_trans"] = Saturation(x0=matrix[var].values[2], alpha=matrix[var].values[3]).fit(
                np.array(df[f"{var}_c"]).reshape(-1, 1)).transform(np.array(df[f"{var}_c"]).reshape(-1, 1))

        model = sm.ols(
            formula=formula(set(list([i + '_trans' for i in matrix.columns if i not in minus_variables]) + context_vars)),
            data=df).fit(method="pinv")

        coef = pd.DataFrame(model.params).T

        coef.to_excel(f'data/interim/step3/coefs_{number}.xlsx', index=False)

        coef_columns_backup = coef.columns.copy()

        coef.columns = [i.replace('_trans', '') for i in coef.columns]

        if number >= 6:
            minus_variables += list(set(coef.columns[(coef < 0).any()].tolist()).intersection(set(paid_vars_imp)))

        ROI = pd.DataFrame([[0, 0]], columns=['index', 0])

        coef.columns = coef_columns_backup

        for var in list(set(matrix.columns) - set(minus_variables)):

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


            elif var == 'nat_tv_wo2020_angus_imp_norm_sov':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['nat_tv_angus_spend'])
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


            elif var == 'digital_2020_2022Q1_imp':
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df['digital_2020_2022Q1_spend'])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))
            else:
                roi = {var: np.sum(df[f"{var}_trans"] * coef[f"{var}_trans"].values) * np.mean(df['avg_check']) \
                            / np.sum(df[var])
                       }
                ROI = ROI.append(pd.DataFrame.from_dict(roi, orient='index').reset_index().fillna(0))

        ROI.columns = ['chanel', 'roi']
        ROI = ROI[1:]

        ROI.to_excel(f'data/interim/step3/roi_{number}.xlsx')

        image = Image.new('RGB', (1200, 800))
        draw = ImageDraw.Draw(image)
        # font = ImageFont.truetype("arial.ttf", 16)
        draw.text((0, 0), str(model.summary()))
        image = image.resize((1200, 800), Image.LANCZOS)
        image.save(f'data/interim/step3/model_{number}.png')

        print(*minus_variables, sep='\n')
        # break

