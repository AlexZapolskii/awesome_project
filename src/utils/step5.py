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
from utils.enums import *
from utils.utils import *



def multindex_iloc(df, index):
    label = df.index.levels[0][index]
    return df.iloc[df.index.get_loc(label)]
class Step5:

    def __init__(self, data):
        comp_budget = 1e9 * 5.5

        for i in (os.listdir('data/interim/step3/')):
            if 'ROI_w_Impact' in i and 'all' not in i:

                df = pd.read_excel('data/interim/step3/' + i)

                if i.split('_w_ROI_w_Impact.xlsx')[0] == 'nat_tv_wo2020_angus_imp_norm_sov':

                    idx = pd.MultiIndex.from_product([range(len(df)), range(len(data))],
                                                     names=['Model', 'Index'])

                    res_idx = pd.MultiIndex.from_product([[0], [0]],
                                                         names=['Model', 'Index'])

                    columns = ['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                               'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                               'x_revenue_cpp', 'y_revenue_roi', \
                               'coef', 'x0', 'alpha', 'strength', 'length', 'avg_check', 'weight']

                    debug_df = pd.DataFrame('-', idx, columns)

                    res_debug_df = pd.DataFrame('-', res_idx, columns)

                    os.mkdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])
                    os.chdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])

                    y = [0] * len(data)
                    y_impact = [0] * len(data)
                    y_adstock = [0] * len(data)
                    y_adstock_saturation = [0] * len(data)
                    y_adstock_saturation_impact = [0] * len(data)
                    y_revenue = [0] * len(data)
                    y_revenue_roi = [0] * len(data)

                    for index, val in df.iterrows():
                        temp_debug_df = multindex_iloc(debug_df, index)

                        coef = val['coef']
                        x0 = val['x0']
                        alpha = val['alpha']
                        strength = val['strength']
                        length = val['length']

                        x_for_graphs = np.linspace(0, int(np.max(data['nat_tv_wo2020_angus_imp_norm'])), int(len(data)))
                        cpp = np.sum(data[data.year >= 2021]['nat_tv_angus_spend']) / np.sum(
                            data['nat_tv_wo2020_angus_imp_norm_sov'])
                        x_for_calcs = ((x_for_graphs / np.mean(
                            data['nat_tv_wo2020_angus_imp_norm_sov'][data['nat_tv_wo2020_angus_imp_norm_sov'] != 0]))) + ((
                                    (x_for_graphs * cpp / (comp_budget / 52 + x_for_graphs)) / np.mean(
                                data['SOV'][data['SOV'] != 0])))

                        y += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        y_impact += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight'] * coef
                        y_adstock += x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        y_adstock_saturation += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight']
                        y_adstock_saturation_impact += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef
                        y_revenue += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check)
                        x_revenue_cpp = x_for_graphs * cpp
                        # x_revenue = x_for_graphs / cpp
                        # x_revenue = ((x_revenue / np.mean(data['nat_tv_wo2020_imp_sov'])) * 0.6) + (((x / (comp_budget / 52 + x)) / np.mean(data['SOV'])) * 0.4)
                        y_revenue_roi += (Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check) / x_revenue_cpp)

                        temp_debug_df['x_for_graphs'] = x_for_graphs
                        temp_debug_df['x_for_calcs'] = x_for_calcs

                        temp_debug_df['y'] = Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        temp_debug_df['y_impact'] = temp_debug_df['y'] * coef
                        temp_debug_df['y_adstock'] = x_for_calcs * (
                                    1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        temp_debug_df['y_adstock_saturation'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight']
                        temp_debug_df['y_adstock_saturation_impact'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef
                        temp_debug_df['y_revenue'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef * np.mean(
                            data.avg_check)
                        temp_debug_df['cpp'] = np.sum(data['nat_tv_angus_spend']) / np.sum(
                            data['nat_tv_wo2020_angus_imp_norm_sov'])
                        temp_debug_df['x_revenue_cpp'] = x_for_graphs * cpp
                        temp_debug_df['y_revenue_roi'] = (temp_debug_df['y_revenue'] / temp_debug_df['x_revenue_cpp'])

                        temp_debug_df['coef'] = coef
                        temp_debug_df['x0'] = x0
                        temp_debug_df['alpha'] = alpha
                        temp_debug_df['strength'] = strength
                        temp_debug_df['length'] = length
                        temp_debug_df['avg_check'] = np.mean(data.avg_check)
                        temp_debug_df['weight'] = val['weight']

                        res_debug_df = res_debug_df.append(temp_debug_df)

                        del temp_debug_df

                    sns.lineplot(x_for_graphs, y)
                    title = f"Weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_impact)
                    title = f"Weighted weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation)
                    title = f"Context curve after ad-stock transform + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation_impact)
                    title = f"Context curve after ad-stock transform with impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue)
                    title = f"Revenue curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('Revenue')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue_roi)
                    title = f"ROI curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('ROI')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    res_debug_df[1:].reset_index().to_excel(f"debug_df_{i.split('_w_ROI_w_Impact.xlsx')[0]}.xlsx",
                                                            index=False)

                    res_x_y = pd.DataFrame(
                        columns=['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                                 'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                                 'x_revenue_cpp', 'y_revenue_roi'])

                    res_x_y['x_for_graphs'] = x_for_graphs
                    res_x_y['x_for_calcs'] = x_for_calcs
                    res_x_y['y'] = y
                    res_x_y['y_impact'] = y_impact
                    res_x_y['y_adstock'] = y_adstock
                    res_x_y['y_adstock_saturation'] = y_adstock_saturation
                    res_x_y['y_adstock_saturation_impact'] = y_adstock_saturation_impact
                    res_x_y['y_revenue'] = y_revenue
                    res_x_y['cpp'] = cpp
                    res_x_y['x_revenue_cpp'] = x_revenue_cpp
                    res_x_y['y_revenue_roi'] = y_revenue_roi

                    res_x_y.to_excel(
                        f"resulting_x_y_{i.split('_w_ROI_w_Impact.xlsx')[0]}_{str(datetime.datetime.now())[:19]}_new.xlsx",
                        index=False)

                    #os.chdir('../')
                    os.chdir('/Users/alexeyzapolskii/PycharmProjects/awesome_project')
                    print(os.getcwd())



                elif i.split('_w_ROI_w_Impact.xlsx')[0] == 'nat_tv_wo2020_product_imp_sov':

                    idx = pd.MultiIndex.from_product([range(len(df)), range(len(data))],
                                                     names=['Model', 'Index'])

                    res_idx = pd.MultiIndex.from_product([[0], [0]],
                                                         names=['Model', 'Index'])

                    columns = ['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                               'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                               'x_revenue_cpp', 'y_revenue_roi', \
                               'coef', 'x0', 'alpha', 'strength', 'length', 'avg_check', 'weight']

                    debug_df = pd.DataFrame('-', idx, columns)

                    res_debug_df = pd.DataFrame('-', res_idx, columns)

                    os.mkdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])
                    os.chdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])

                    y = [0] * len(data)
                    y_impact = [0] * len(data)
                    y_adstock = [0] * len(data)
                    y_adstock_saturation = [0] * len(data)
                    y_adstock_saturation_impact = [0] * len(data)
                    y_revenue = [0] * len(data)
                    y_revenue_roi = [0] * len(data)

                    for index, val in df.iterrows():
                        temp_debug_df = multindex_iloc(debug_df, index)

                        coef = val['coef']
                        x0 = val['x0']
                        alpha = val['alpha']
                        strength = val['strength']
                        length = val['length']

                        #                 x_for_graphs = np.linspace(0, int(np.max(data[['nat_tv_product_imp']].reset_index()[data[['nat_tv_product_imp']].reset_index()['date'].dt.year >= 2021]['nat_tv_product_imp'])), int(len(data)))
                        #                 cpp = np.sum(data[data.year >= 2021]['nat_tv_product_spend']) / np.sum(data['nat_tv_product_imp'])
                        #                 x_for_calcs = \
                        #                    (0.5 * ((x_for_graphs / np.mean(df[['nat_tv_product_imp']].reset_index()[(df[['nat_tv_product_imp']].reset_index()['date'].dt.year >= 2021) & \
                        #                                          (df[['nat_tv_product_imp']].reset_index()['nat_tv_product_imp']) != 0 ]['nat_tv_product_imp'])))) \
                        #                     + (0.5 * (((x_for_graphs * cpp / (comp_budget / 52 + x_for_graphs )) / np.mean(df[['SOV_product']].reset_index()[(df[['SOV_product']].reset_index()['date'].dt.year >= 2021) & \
                        #                                          (df[['SOV_product']].reset_index()['SOV_product']) != 0 ]['SOV_product']))))

                        x_for_graphs = np.linspace(0, int(np.max(data[data.year >= 2021]['nat_tv_product_imp'])),
                                                   int(len(data)))
                        cpp = np.sum(data[data.year >= 2021]['nat_tv_product_spend']) / np.sum(data['nat_tv_product_imp'])
                        x_for_calcs = \
                            (0.5 * ((x_for_graphs / np.mean(
                                data[(data.year >= 2021) & (data.nat_tv_product_imp != 0)]['nat_tv_product_imp'])))) \
                            + (0.5 * (((x_for_graphs * cpp / (comp_budget / 52 + x_for_graphs)) / np.mean(
                                data[(data.year >= 2021) & (data.SOV_product != 0)]['SOV_product']))))

                        y += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        y_impact += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight'] * coef
                        y_adstock += x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        y_adstock_saturation += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight']
                        y_adstock_saturation_impact += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef
                        y_revenue += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check)
                        x_revenue_cpp = x_for_graphs * cpp
                        # x_revenue = x_for_graphs / cpp
                        # x_revenue = ((x_revenue / np.mean(data['nat_tv_wo2020_imp_sov'])) * 0.6) + (((x / (comp_budget / 52 + x)) / np.mean(data['SOV'])) * 0.4)
                        y_revenue_roi += (Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check) / x_revenue_cpp)

                        temp_debug_df['x_for_graphs'] = x_for_graphs
                        temp_debug_df['x_for_calcs'] = x_for_calcs

                        temp_debug_df['y'] = Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        temp_debug_df['y_impact'] = temp_debug_df['y'] * coef
                        temp_debug_df['y_adstock'] = x_for_calcs * (
                                    1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        temp_debug_df['y_adstock_saturation'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight']
                        temp_debug_df['y_adstock_saturation_impact'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef
                        temp_debug_df['y_revenue'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef * np.mean(
                            data.avg_check)
                        temp_debug_df['cpp'] = np.sum(data['nat_tv_product_spend']) / np.sum(
                            data['nat_tv_wo2020_product_imp_sov'])
                        temp_debug_df['x_revenue_cpp'] = x_for_graphs * cpp
                        temp_debug_df['y_revenue_roi'] = (temp_debug_df['y_revenue'] / temp_debug_df['x_revenue_cpp'])

                        temp_debug_df['coef'] = coef
                        temp_debug_df['x0'] = x0
                        temp_debug_df['alpha'] = alpha
                        temp_debug_df['strength'] = strength
                        temp_debug_df['length'] = length
                        temp_debug_df['avg_check'] = np.mean(data.avg_check)
                        temp_debug_df['weight'] = val['weight']

                        res_debug_df = res_debug_df.append(temp_debug_df)

                        del temp_debug_df

                    sns.lineplot(x_for_graphs, y)
                    title = f"Weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_impact)
                    title = f"Weighted weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation)
                    title = f"Context curve after ad-stock transform + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation_impact)
                    title = f"Context curve after ad-stock transform with impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue)
                    title = f"Revenue curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('Revenue')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue_roi)
                    title = f"ROI curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('ROI')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    res_debug_df[1:].reset_index().to_excel(f"debug_df_{i.split('_w_ROI_w_Impact.xlsx')[0]}.xlsx",
                                                            index=False)

                    res_x_y = pd.DataFrame(
                        columns=['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                                 'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                                 'x_revenue_cpp', 'y_revenue_roi'])

                    res_x_y['x_for_graphs'] = x_for_graphs
                    res_x_y['x_for_calcs'] = x_for_calcs
                    res_x_y['y'] = y
                    res_x_y['y_impact'] = y_impact
                    res_x_y['y_adstock'] = y_adstock
                    res_x_y['y_adstock_saturation'] = y_adstock_saturation
                    res_x_y['y_adstock_saturation_impact'] = y_adstock_saturation_impact
                    res_x_y['y_revenue'] = y_revenue
                    res_x_y['cpp'] = cpp
                    res_x_y['x_revenue_cpp'] = x_revenue_cpp
                    res_x_y['y_revenue_roi'] = y_revenue_roi

                    res_x_y.to_excel(
                        f"resulting_x_y_{i.split('_w_ROI_w_Impact.xlsx')[0]}_{str(datetime.datetime.now())[:19]}_new.xlsx",
                        index=False)

                    print(os.getcwd())

                    #os.chdir('../')

                    os.chdir('/Users/alexeyzapolskii/PycharmProjects/awesome_project')

                    print(os.getcwd())


                elif i.split('_w_ROI_w_Impact.xlsx')[0] == 'nat_tv_wo2020_vfm_imp_sov':

                    idx = pd.MultiIndex.from_product([range(len(df)), range(len(data))],
                                                     names=['Model', 'Index'])

                    res_idx = pd.MultiIndex.from_product([[0], [0]],
                                                         names=['Model', 'Index'])

                    columns = ['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                               'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                               'x_revenue_cpp', 'y_revenue_roi', \
                               'coef', 'x0', 'alpha', 'strength', 'length', 'avg_check', 'weight']

                    debug_df = pd.DataFrame('-', idx, columns)

                    res_debug_df = pd.DataFrame('-', res_idx, columns)

                    os.mkdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])
                    os.chdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])

                    y = [0] * len(data)
                    y_impact = [0] * len(data)
                    y_adstock = [0] * len(data)
                    y_adstock_saturation = [0] * len(data)
                    y_adstock_saturation_impact = [0] * len(data)
                    y_revenue = [0] * len(data)
                    y_revenue_roi = [0] * len(data)

                    for index, val in df.iterrows():
                        temp_debug_df = multindex_iloc(debug_df, index)

                        coef = val['coef']
                        x0 = val['x0']
                        alpha = val['alpha']
                        strength = val['strength']
                        length = val['length']

                        #                 x_for_graphs = np.linspace(0, int(np.max(df[['nat_tv_vfm_imp']].reset_index()[df[['nat_tv_vfm_imp']].reset_index()['date'].dt.year >= 2021]['nat_tv_vfm_imp'])), int(len(data)))
                        #                 cpp = np.sum(data[data.year >= 2021]['nat_tv_vfm_spend']) / np.sum(data['nat_tv_vfm_imp'])
                        #                 x_for_calcs = \
                        #                    (0.5 * ((x_for_graphs / np.mean(df[['nat_tv_vfm_imp']].reset_index()[(df[['nat_tv_vfm_imp']].reset_index()['date'].dt.year >= 2021) & \
                        #                                          (df[['nat_tv_vfm_imp']].reset_index()['nat_tv_vfm_imp']) != 0 ]['nat_tv_vfm_imp'])))) \
                        #                     + (0.5 * (((x_for_graphs * cpp / (comp_budget / 52 + x_for_graphs )) / np.mean(df[['SOV_vfm']].reset_index()[(df[['SOV_vfm']].reset_index()['date'].dt.year >= 2021) & \
                        #                                          (df[['SOV_vfm']].reset_index()['SOV_vfm']) != 0 ]['SOV_vfm']))))

                        x_for_graphs = np.linspace(0, int(np.max(data[data.year >= 2021]['nat_tv_vfm_imp'])),
                                                   int(len(data)))
                        cpp = np.sum(data[data.year >= 2021]['nat_tv_vfm_spend']) / np.sum(data['nat_tv_vfm_imp'])
                        x_for_calcs = \
                            (0.5 * ((x_for_graphs / np.mean(
                                data[(data.year >= 2021) & (data.nat_tv_vfm_imp != 0)]['nat_tv_vfm_imp'])))) \
                            + (0.5 * (((x_for_graphs * cpp / (comp_budget / 52 + x_for_graphs)) / np.mean(
                                data[(data.year >= 2021) & (data.SOV_vfm != 0)]['SOV_vfm']))))

                        y += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        y_impact += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight'] * coef
                        y_adstock += x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        y_adstock_saturation += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight']
                        y_adstock_saturation_impact += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef
                        y_revenue += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check)
                        x_revenue_cpp = x_for_graphs * cpp
                        # x_revenue = x_for_graphs / cpp
                        # x_revenue = ((x_revenue / np.mean(data['nat_tv_wo2020_imp_sov'])) * 0.6) + (((x / (comp_budget / 52 + x)) / np.mean(data['SOV'])) * 0.4)
                        y_revenue_roi += (Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check) / x_revenue_cpp)

                        temp_debug_df['x_for_graphs'] = x_for_graphs
                        temp_debug_df['x_for_calcs'] = x_for_calcs

                        temp_debug_df['y'] = Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        temp_debug_df['y_impact'] = temp_debug_df['y'] * coef
                        temp_debug_df['y_adstock'] = x_for_calcs * (
                                    1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        temp_debug_df['y_adstock_saturation'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight']
                        temp_debug_df['y_adstock_saturation_impact'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef
                        temp_debug_df['y_revenue'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef * np.mean(
                            data.avg_check)
                        temp_debug_df['cpp'] = np.sum(data['nat_tv_vfm_spend']) / np.sum(data['nat_tv_wo2020_vfm_imp_sov'])
                        temp_debug_df['x_revenue_cpp'] = x_for_graphs * cpp
                        temp_debug_df['y_revenue_roi'] = (temp_debug_df['y_revenue'] / temp_debug_df['x_revenue_cpp'])

                        temp_debug_df['coef'] = coef
                        temp_debug_df['x0'] = x0
                        temp_debug_df['alpha'] = alpha
                        temp_debug_df['strength'] = strength
                        temp_debug_df['length'] = length
                        temp_debug_df['avg_check'] = np.mean(data.avg_check)
                        temp_debug_df['weight'] = val['weight']

                        res_debug_df = res_debug_df.append(temp_debug_df)

                        del temp_debug_df

                    sns.lineplot(x_for_graphs, y)
                    title = f"Weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_impact)
                    title = f"Weighted weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation)
                    title = f"Context curve after ad-stock transform + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation_impact)
                    title = f"Context curve after ad-stock transform with impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue)
                    title = f"Revenue curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('Revenue')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue_roi)
                    title = f"ROI curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('ROI')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    res_debug_df[1:].reset_index().to_excel(f"debug_df_{i.split('_w_ROI_w_Impact.xlsx')[0]}.xlsx",
                                                            index=False)

                    res_x_y = pd.DataFrame(
                        columns=['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                                 'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                                 'x_revenue_cpp', 'y_revenue_roi'])

                    res_x_y['x_for_graphs'] = x_for_graphs
                    res_x_y['x_for_calcs'] = x_for_calcs
                    res_x_y['y'] = y
                    res_x_y['y_impact'] = y_impact
                    res_x_y['y_adstock'] = y_adstock
                    res_x_y['y_adstock_saturation'] = y_adstock_saturation
                    res_x_y['y_adstock_saturation_impact'] = y_adstock_saturation_impact
                    res_x_y['y_revenue'] = y_revenue
                    res_x_y['cpp'] = cpp
                    res_x_y['x_revenue_cpp'] = x_revenue_cpp
                    res_x_y['y_revenue_roi'] = y_revenue_roi

                    res_x_y.to_excel(
                        f"resulting_x_y_{i.split('_w_ROI_w_Impact.xlsx')[0]}_{str(datetime.datetime.now())[:19]}_new.xlsx",
                        index=False)

                    #os.chdir('../')
                    os.chdir('/Users/alexeyzapolskii/PycharmProjects/awesome_project')

                    print(os.getcwd())


                elif i.split('_w_ROI_w_Impact.xlsx')[0] == 'digital_none_youtube_imp':

                    idx = pd.MultiIndex.from_product([range(len(df)), range(len(data))],
                                                     names=['Model', 'Index'])

                    res_idx = pd.MultiIndex.from_product([[0], [0]],
                                                         names=['Model', 'Index'])

                    columns = ['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                               'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                               'x_revenue_cpp', 'y_revenue_roi', \
                               'coef', 'x0', 'alpha', 'strength', 'length', 'avg_check', 'weight']

                    debug_df = pd.DataFrame('-', idx, columns)

                    res_debug_df = pd.DataFrame('-', res_idx, columns)

                    os.mkdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])
                    os.chdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])

                    y = [0] * len(data)
                    y_impact = [0] * len(data)
                    y_adstock = [0] * len(data)
                    y_adstock_saturation = [0] * len(data)
                    y_adstock_saturation_impact = [0] * len(data)
                    y_revenue = [0] * len(data)
                    y_revenue_roi = [0] * len(data)

                    for index, val in df.iterrows():
                        temp_debug_df = multindex_iloc(debug_df, index)

                        coef = val['coef']
                        x0 = val['x0']
                        alpha = val['alpha']
                        strength = val['strength']
                        length = val['length']

                        x_for_graphs = np.linspace(0, int(np.max(data[i.split('_w_ROI_w_Impact.xlsx')[0]])), int(len(data)))
                        cpp = np.sum(data['digital_none_youtube_spend']) / np.sum(data[i.split('_w_ROI_w_Impact.xlsx')[0]])
                        x_for_calcs = np.linspace(0, int(np.max(data[i.split('_w_ROI_w_Impact.xlsx')[0]])), int(len(data)))

                        y += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        y_impact += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight'] * coef
                        y_adstock += x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        y_adstock_saturation += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight']
                        y_adstock_saturation_impact += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef
                        y_revenue += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check)
                        x_revenue_cpp = x_for_graphs * cpp
                        y_revenue_roi += (Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check) / x_revenue_cpp)

                        temp_debug_df['x_for_graphs'] = x_for_graphs
                        temp_debug_df['x_for_calcs'] = x_for_calcs

                        temp_debug_df['y'] = Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        temp_debug_df['y_impact'] = temp_debug_df['y'] * coef
                        temp_debug_df['y_adstock'] = x_for_calcs * (
                                    1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        temp_debug_df['y_adstock_saturation'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight']
                        temp_debug_df['y_adstock_saturation_impact'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef
                        temp_debug_df['y_revenue'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef * np.mean(
                            data.avg_check)
                        temp_debug_df['cpp'] = np.sum(
                            data[(data.year == 2022) & (data.month >= 4)]['digital_none_youtube_spend']) / np.sum(
                            data[i.split('_w_ROI_w_Impact.xlsx')[0]])
                        temp_debug_df['x_revenue_cpp'] = x_for_graphs * cpp
                        temp_debug_df['y_revenue_roi'] = (temp_debug_df['y_revenue'] / temp_debug_df['x_revenue_cpp'])

                        temp_debug_df['coef'] = coef
                        temp_debug_df['x0'] = x0
                        temp_debug_df['alpha'] = alpha
                        temp_debug_df['strength'] = strength
                        temp_debug_df['length'] = length
                        temp_debug_df['avg_check'] = np.mean(data.avg_check)
                        temp_debug_df['weight'] = val['weight']

                        res_debug_df = res_debug_df.append(temp_debug_df)

                        del temp_debug_df

                    sns.lineplot(x_for_graphs, y)
                    title = f"Weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_impact)
                    title = f"Weighted weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation)
                    title = f"Context curve after ad-stock transform + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation_impact)
                    title = f"Context curve after ad-stock transform with impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue)
                    title = f"Revenue curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('Revenue')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue_roi)
                    title = f"ROI curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('ROI')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    res_debug_df[1:].reset_index().to_excel(f"debug_{i}.xlsx", index=False)

                    res_x_y = pd.DataFrame(
                        columns=['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                                 'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                                 'x_revenue_cpp', 'y_revenue_roi'])

                    res_x_y['x_for_graphs'] = x_for_graphs
                    res_x_y['x_for_calcs'] = x_for_calcs
                    res_x_y['y'] = y
                    res_x_y['y_impact'] = y_impact
                    res_x_y['y_adstock'] = y_adstock
                    res_x_y['y_adstock_saturation'] = y_adstock_saturation
                    res_x_y['y_adstock_saturation_impact'] = y_adstock_saturation_impact
                    res_x_y['y_revenue'] = y_revenue
                    res_x_y['cpp'] = cpp
                    res_x_y['x_revenue_cpp'] = x_revenue_cpp
                    res_x_y['y_revenue_roi'] = y_revenue_roi

                    res_x_y.to_excel(f"resulting_x_y_{i}_{str(datetime.datetime.now())[:19]}_new.xlsx", index=False)

                    #os.chdir('../')

                    os.chdir('/Users/alexeyzapolskii/PycharmProjects/awesome_project')

                    print(os.getcwd())


                elif i.split('_w_ROI_w_Impact.xlsx')[0] == 'digital_2020_2022Q1_imp':

                    idx = pd.MultiIndex.from_product([range(len(df)), range(len(data))],
                                                     names=['Model', 'Index'])

                    res_idx = pd.MultiIndex.from_product([[0], [0]],
                                                         names=['Model', 'Index'])

                    columns = ['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                               'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                               'x_revenue_cpp', 'y_revenue_roi', \
                               'coef', 'x0', 'alpha', 'strength', 'length', 'avg_check', 'weight']

                    debug_df = pd.DataFrame('-', idx, columns)

                    res_debug_df = pd.DataFrame('-', res_idx, columns)

                    os.mkdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])
                    os.chdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])

                    y = [0] * len(data)
                    y_impact = [0] * len(data)
                    y_adstock = [0] * len(data)
                    y_adstock_saturation = [0] * len(data)
                    y_adstock_saturation_impact = [0] * len(data)
                    y_revenue = [0] * len(data)
                    y_revenue_roi = [0] * len(data)

                    for index, val in df.iterrows():
                        temp_debug_df = multindex_iloc(debug_df, index)

                        coef = val['coef']
                        x0 = val['x0']
                        alpha = val['alpha']
                        strength = val['strength']
                        length = val['length']

                        x_for_graphs = np.linspace(0, int(np.max(data[i.split('_w_ROI_w_Impact.xlsx')[0]])), int(len(data)))
                        cpp = np.sum(data['digital_2020_2022Q1_spend']) / np.sum(data[i.split('_w_ROI_w_Impact.xlsx')[0]])
                        x_for_calcs = np.linspace(0, int(np.max(data[i.split('_w_ROI_w_Impact.xlsx')[0]])), int(len(data)))

                        y += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        y_impact += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight'] * coef
                        y_adstock += x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        y_adstock_saturation += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight']
                        y_adstock_saturation_impact += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef
                        y_revenue += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check)
                        x_revenue_cpp = x_for_graphs * cpp
                        y_revenue_roi += (Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check) / x_revenue_cpp)

                        temp_debug_df['x_for_graphs'] = x_for_graphs
                        temp_debug_df['x_for_calcs'] = x_for_calcs

                        temp_debug_df['y'] = Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        temp_debug_df['y_impact'] = temp_debug_df['y'] * coef
                        temp_debug_df['y_adstock'] = x_for_calcs * (
                                    1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        temp_debug_df['y_adstock_saturation'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight']
                        temp_debug_df['y_adstock_saturation_impact'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef
                        temp_debug_df['y_revenue'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef * np.mean(
                            data.avg_check)
                        temp_debug_df['cpp'] = np.sum(data['digital_2020_2022Q1_spend']) / np.sum(
                            data[i.split('_w_ROI_w_Impact.xlsx')[0]])
                        temp_debug_df['x_revenue_cpp'] = x_for_graphs * cpp
                        temp_debug_df['y_revenue_roi'] = (temp_debug_df['y_revenue'] / temp_debug_df['x_revenue_cpp'])

                        temp_debug_df['coef'] = coef
                        temp_debug_df['x0'] = x0
                        temp_debug_df['alpha'] = alpha
                        temp_debug_df['strength'] = strength
                        temp_debug_df['length'] = length
                        temp_debug_df['avg_check'] = np.mean(data.avg_check)
                        temp_debug_df['weight'] = val['weight']

                        res_debug_df = res_debug_df.append(temp_debug_df)

                        del temp_debug_df

                    sns.lineplot(x_for_graphs, y)
                    title = f"Weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_impact)
                    title = f"Weighted weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation)
                    title = f"Context curve after ad-stock transform + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation_impact)
                    title = f"Context curve after ad-stock transform with impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue)
                    title = f"Revenue curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('Revenue')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue_roi)
                    title = f"ROI curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('ROI')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    res_debug_df[1:].reset_index().to_excel(f"debug_{i}.xlsx", index=False)

                    res_x_y = pd.DataFrame(
                        columns=['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                                 'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                                 'x_revenue_cpp', 'y_revenue_roi'])

                    res_x_y['x_for_graphs'] = x_for_graphs
                    res_x_y['x_for_calcs'] = x_for_calcs
                    res_x_y['y'] = y
                    res_x_y['y_impact'] = y_impact
                    res_x_y['y_adstock'] = y_adstock
                    res_x_y['y_adstock_saturation'] = y_adstock_saturation
                    res_x_y['y_adstock_saturation_impact'] = y_adstock_saturation_impact
                    res_x_y['y_revenue'] = y_revenue
                    res_x_y['cpp'] = cpp
                    res_x_y['x_revenue_cpp'] = x_revenue_cpp
                    res_x_y['y_revenue_roi'] = y_revenue_roi

                    res_x_y.to_excel(f"resulting_x_y_{i}_{str(datetime.datetime.now())[:19]}_new.xlsx", index=False)

                    #os.chdir('../')
                    os.chdir('/Users/alexeyzapolskii/PycharmProjects/awesome_project')

                    print(os.getcwd())

                elif i.split('_w_ROI_w_Impact.xlsx')[0] not in ['digital_none_youtube_imp', 'nat_tv_wo2020_imp_sov',
                                                                'digital_2020_2022Q1_imp'] and 'imp' not in \
                        i.split('_w_ROI_w_Impact.xlsx')[0] and i.split('_w_ROI_w_Impact.xlsx')[0] in paid_vars_imp and \
                        i.split('_w_ROI_w_Impact.xlsx')[0] not in ['digital_none_youtube_spend', 'nat_tv_wo2020_imp_sov']:

                    os.mkdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])
                    os.chdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])

                    idx = pd.MultiIndex.from_product([range(len(df)), range(len(data))],
                                                     names=['Model', 'Index'])

                    res_idx = pd.MultiIndex.from_product([[0], [0]],
                                                         names=['Model', 'Index'])

                    columns = ['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                               'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                               'x_revenue_cpp', 'y_revenue_roi', \
                               'coef', 'x0', 'alpha', 'strength', 'length', 'avg_check', 'weight']

                    debug_df = pd.DataFrame('-', idx, columns)

                    res_debug_df = pd.DataFrame('-', res_idx, columns)

                    y = [0] * len(data)
                    y_impact = [0] * len(data)
                    y_adstock = [0] * len(data)
                    y_adstock_saturation = [0] * len(data)
                    y_adstock_saturation_impact = [0] * len(data)
                    y_revenue = [0] * len(data)
                    y_revenue_roi = [0] * len(data)

                    for index, val in df.iterrows():
                        temp_debug_df = multindex_iloc(debug_df, index)

                        coef = val['coef']
                        x0 = val['x0']
                        alpha = val['alpha']
                        strength = val['strength']
                        length = val['length']

                        x_for_graphs = np.linspace(0, int(np.max(data[i.split('_w_ROI_w_Impact.xlsx')[0]])), int(len(data)))
                        cpp = np.sum(data[i.split('_w_ROI_w_Impact.xlsx')[0]]) / np.sum(
                            data[i.split('_w_ROI_w_Impact.xlsx')[0]])
                        x_for_calcs = np.linspace(0, int(np.max(data[i.split('_w_ROI_w_Impact.xlsx')[0]])), int(len(data)))

                        y += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        y_impact += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight'] * coef
                        y_adstock += x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        y_adstock_saturation += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight']
                        y_adstock_saturation_impact += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef
                        y_revenue += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check)
                        x_revenue_cpp = x_for_graphs * cpp
                        y_revenue_roi += (Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check) / x_revenue_cpp)

                        temp_debug_df['x_for_graphs'] = x_for_graphs
                        temp_debug_df['x_for_calcs'] = x_for_calcs

                        temp_debug_df['y'] = Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        temp_debug_df['y_impact'] = temp_debug_df['y'] * coef
                        temp_debug_df['y_adstock'] = x_for_calcs * (
                                    1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        temp_debug_df['y_adstock_saturation'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight']
                        temp_debug_df['y_adstock_saturation_impact'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef
                        temp_debug_df['y_revenue'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef * np.mean(
                            data.avg_check)
                        temp_debug_df['cpp'] = np.sum(data[i.split('_w_ROI_w_Impact.xlsx')[0]]) / np.sum(
                            data[i.split('_w_ROI_w_Impact.xlsx')[0]])
                        temp_debug_df['x_revenue_cpp'] = x_for_graphs * cpp
                        temp_debug_df['y_revenue_roi'] = (temp_debug_df['y_revenue'] / temp_debug_df['x_revenue_cpp'])

                        temp_debug_df['coef'] = coef
                        temp_debug_df['x0'] = x0
                        temp_debug_df['alpha'] = alpha
                        temp_debug_df['strength'] = strength
                        temp_debug_df['length'] = length
                        temp_debug_df['avg_check'] = np.mean(data.avg_check)
                        temp_debug_df['weight'] = val['weight']

                        res_debug_df = res_debug_df.append(temp_debug_df)

                        del temp_debug_df

                    sns.lineplot(x_for_graphs, y)
                    title = f"Weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_impact)
                    title = f"Weighted weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation)
                    title = f"Context curve after ad-stock transform + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation_impact)
                    title = f"Context curve after ad-stock transform with impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue)
                    title = f"Revenue curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('Revenue')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue_roi)
                    title = f"ROI curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('ROI')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    res_debug_df[1:].reset_index().to_excel(f"debug_{i}.xlsx", index=False)

                    res_x_y = pd.DataFrame(
                        columns=['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                                 'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                                 'x_revenue_cpp', 'y_revenue_roi'])

                    res_x_y['x_for_graphs'] = x_for_graphs
                    res_x_y['x_for_calcs'] = x_for_calcs
                    res_x_y['y'] = y
                    res_x_y['y_impact'] = y_impact
                    res_x_y['y_adstock'] = y_adstock
                    res_x_y['y_adstock_saturation'] = y_adstock_saturation
                    res_x_y['y_adstock_saturation_impact'] = y_adstock_saturation_impact
                    res_x_y['y_revenue'] = y_revenue
                    res_x_y['cpp'] = cpp
                    res_x_y['x_revenue_cpp'] = x_revenue_cpp
                    res_x_y['y_revenue_roi'] = y_revenue_roi

                    res_x_y.to_excel(f"resulting_x_y_{i}_{str(datetime.datetime.now())[:19]}_new.xlsx", index=False)

                    #os.chdir('../')
                    os.chdir('/Users/alexeyzapolskii/PycharmProjects/awesome_project')

                    print(os.getcwd())


                elif i.split('_w_ROI_w_Impact.xlsx')[0] not in ['digital_none_youtube_imp',
                                                                'nat_tv_wo2020_imp_sov'] and 'imp' in \
                        i.split('_w_ROI_w_Impact.xlsx')[0] and i.split('_w_ROI_w_Impact.xlsx')[0] in paid_vars_imp:

                    os.mkdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])
                    os.chdir('data/interim/step3/' + i.split('_w_ROI_w_Impact.xlsx')[0])

                    idx = pd.MultiIndex.from_product([range(len(df)), range(len(data))],
                                                     names=['Model', 'Index'])

                    res_idx = pd.MultiIndex.from_product([[0], [0]],
                                                         names=['Model', 'Index'])

                    columns = ['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                               'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                               'x_revenue_cpp', 'y_revenue_roi', \
                               'coef', 'x0', 'alpha', 'strength', 'length', 'avg_check', 'weight']

                    debug_df = pd.DataFrame('-', idx, columns)

                    res_debug_df = pd.DataFrame('-', res_idx, columns)

                    y = [0] * len(data)
                    y_impact = [0] * len(data)
                    y_adstock = [0] * len(data)
                    y_adstock_saturation = [0] * len(data)
                    y_adstock_saturation_impact = [0] * len(data)
                    y_revenue = [0] * len(data)
                    y_revenue_roi = [0] * len(data)

                    for index, val in df.iterrows():
                        temp_debug_df = multindex_iloc(debug_df, index)

                        coef = val['coef']
                        x0 = val['x0']
                        alpha = val['alpha']
                        strength = val['strength']
                        length = val['length']

                        x_for_graphs = np.linspace(0, int(np.max(data[i.split('_w_ROI_w_Impact.xlsx')[0]])), int(len(data)))
                        cpp = np.sum(data[i.split('_w_ROI_w_Impact.xlsx')[0].replace('imp', 'spend')]) / np.sum(
                            data[i.split('_w_ROI_w_Impact.xlsx')[0]])
                        x_for_calcs = np.linspace(0, int(np.max(data[i.split('_w_ROI_w_Impact.xlsx')[0]])), int(len(data)))

                        y += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        y_impact += Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight'] * coef
                        y_adstock += x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        y_adstock_saturation += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight']
                        y_adstock_saturation_impact += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef
                        y_revenue += Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check)
                        x_revenue_cpp = x_for_graphs * cpp
                        y_revenue_roi += (Saturation(x0=x0, alpha=alpha).fit(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).transform(
                            x_for_calcs * (1 + np.sum([strength ** i for i in range(1, length + 1)])).reshape(-1,
                                                                                                              1)).reshape(
                            -1) * val['weight'] * coef * np.mean(data.avg_check) / x_revenue_cpp)

                        temp_debug_df['x_for_graphs'] = x_for_graphs
                        temp_debug_df['x_for_calcs'] = x_for_calcs

                        temp_debug_df['y'] = Saturation(x0=x0, alpha=alpha).fit(x_for_calcs.reshape(-1, 1)).transform(
                            x_for_calcs.reshape(-1, 1)).reshape(-1) * val['weight']
                        temp_debug_df['y_impact'] = temp_debug_df['y'] * coef
                        temp_debug_df['y_adstock'] = x_for_calcs * (
                                    1 + np.sum([strength ** i for i in range(1, length + 1)]))
                        temp_debug_df['y_adstock_saturation'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight']
                        temp_debug_df['y_adstock_saturation_impact'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef
                        temp_debug_df['y_revenue'] = (Saturation(x0=x0, alpha=alpha).fit(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).transform(
                            temp_debug_df['y_adstock'].values.reshape(-1, 1)).reshape(-1)) * val['weight'] * coef * np.mean(
                            data.avg_check)
                        temp_debug_df['cpp'] = np.sum(data[i.split('_w_ROI_w_Impact.xlsx')[0]]) / np.sum(
                            data[i.split('_w_ROI_w_Impact.xlsx')[0]])
                        temp_debug_df['x_revenue_cpp'] = x_for_graphs * cpp
                        temp_debug_df['y_revenue_roi'] = (temp_debug_df['y_revenue'] / temp_debug_df['x_revenue_cpp'])

                        temp_debug_df['coef'] = coef
                        temp_debug_df['x0'] = x0
                        temp_debug_df['alpha'] = alpha
                        temp_debug_df['strength'] = strength
                        temp_debug_df['length'] = length
                        temp_debug_df['avg_check'] = np.mean(data.avg_check)
                        temp_debug_df['weight'] = val['weight']

                        res_debug_df = res_debug_df.append(temp_debug_df)

                        del temp_debug_df

                    sns.lineplot(x_for_graphs, y)
                    title = f"Weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_impact)
                    title = f"Weighted weekly incremental impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation)
                    title = f"Context curve after ad-stock transform + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_for_graphs, y_adstock_saturation_impact)
                    title = f"Context curve after ad-stock transform with impact + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends')
                    plt.ylabel('Impact')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue)
                    title = f"Revenue curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('Revenue')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    sns.lineplot(x_revenue_cpp, y_revenue_roi)
                    title = f"ROI curve after ad-stock by CPP + {i.split('_w_ROI_w_Impact.xlsx')[0]}"
                    plt.title(title)
                    plt.xlabel('Spends i.e. CPP')
                    plt.ylabel('ROI')
                    plt.grid()
                    plt.savefig(title + '.png', dpi=200)
                    plt.close()

                    res_debug_df[1:].reset_index().to_excel(f"debug_{i}.xlsx", index=False)

                    res_x_y = pd.DataFrame(
                        columns=['x_for_graphs', 'x_for_calcs', 'y', 'y_impact', 'y_adstock', 'y_adstock_saturation', \
                                 'y_adstock_saturation_impact', 'y_revenue', 'cpp', \
                                 'x_revenue_cpp', 'y_revenue_roi'])

                    res_x_y['x_for_graphs'] = x_for_graphs
                    res_x_y['x_for_calcs'] = x_for_calcs
                    res_x_y['y'] = y
                    res_x_y['y_impact'] = y_impact
                    res_x_y['y_adstock'] = y_adstock
                    res_x_y['y_adstock_saturation'] = y_adstock_saturation
                    res_x_y['y_adstock_saturation_impact'] = y_adstock_saturation_impact
                    res_x_y['y_revenue'] = y_revenue
                    res_x_y['cpp'] = cpp
                    res_x_y['x_revenue_cpp'] = x_revenue_cpp
                    res_x_y['y_revenue_roi'] = y_revenue_roi

                    res_x_y.to_excel(f"resulting_x_y_{i}_{str(datetime.datetime.now())[:19]}_new.xlsx", index=False)

                    #os.chdir('../')
                    os.chdir('/Users/alexeyzapolskii/PycharmProjects/awesome_project')

                    print(os.getcwd())

            return None

    def process_data(self, data):

        temp_tab = pd.DataFrame(columns=['weighted_roi_2020', 'weighted_impact_2020', \
                                         'weighted_roi_2021', 'weighted_impact_2021', \
                                         'weighted_roi_2022_Q1', 'weighted_impact_2022_Q1', \
                                         'weighted_roi_2022_Q2', 'weighted_impact_2022_Q2', \
                                         'weighted_roi_2022_Q3', 'weighted_impact_2022_Q3', \
                                         'weighted_roi_2022_Q4', 'weighted_impact_2022_Q4', \
                                         'weighted_roi_2023_Q1', 'weighted_impact_2023_Q1'])

        for file in [i for i in os.listdir('data/interim/step3/') if 'ROI_w_Impact' in i and 'all' not in i]:
            temp_df = pd.read_excel('data/interim/step3/' + file)
            temp_res = pd.DataFrame(temp_df[['weighted_roi_2020', 'weighted_impact_2020', \
                                             'weighted_roi_2021', 'weighted_impact_2021', \
                                             'weighted_roi_2022_Q1', 'weighted_impact_2022_Q1', \
                                             'weighted_roi_2022_Q2', 'weighted_impact_2022_Q2', \
                                             'weighted_roi_2022_Q3', 'weighted_impact_2022_Q3', \
                                             'weighted_roi_2022_Q4', 'weighted_impact_2022_Q4', \
                                             'weighted_roi_2023_Q1', 'weighted_impact_2023_Q1']].iloc[0, :]).T
            temp_res = temp_res.rename(index={1: f"{file.split('_w_ROI_w_Impact.xlsx')[0]}"})
            temp_tab = temp_tab.append(temp_res)

        temp_tab = temp_tab.reset_index()
        temp_tab['index'] = [i.split('_w_ROI_w_Impact.xlsx')[0] for i in os.listdir('data/interim/step3/') if
                             'ROI_w_Impact' in i and 'all' not in i]
        temp_tab = temp_tab.set_index('index')

        temp_tab[[i for i in temp_tab.columns if 'roi' in i]].to_excel('data/interim/step3/' + 'roi_breakdown.xlsx')
        temp_tab[[i for i in temp_tab.columns if 'roi' in i]]

        # читаем веса последней модели
        latest_weights = pd.read_excel('data/interim/step3/' + "coefs_11.xlsx")

        # снова data! правильное использование???
        context_regressors = data.copy()

        context_regressors = context_regressors[context_vars].mul(latest_weights[context_vars].iloc[0].values)

        context_regressors.to_excel('data/interim/step3/' +'context_regressors.xlsx')

        for i in (os.listdir('data/interim/step3/')):

            if 'ROI_w_Impact' in i and 'all' not in i:

                columns = ['coef', 'x0', 'alpha', 'strength', 'length', 'weight']

                res_idx = pd.MultiIndex.from_product([[0], [0]], names=['Model', 'Index'])

                res_debug_df = pd.DataFrame('-', res_idx, columns)

                df = pd.read_excel('data/interim/step3/' + i).fillna(0)

                idx = pd.MultiIndex.from_product([range(len(df)), range(len(data))],
                                                 names=['Model', 'Index'])

                marketing_var = i.split('_w_ROI_w_Impact.xlsx')[0]

                columns = ['coef', 'x0', 'alpha', 'strength', 'length', 'weight']

                debug_df = pd.DataFrame('-', idx, columns)

                for index, values in df.iterrows():
                    temp_debug_df = multindex_iloc(debug_df, index).reset_index()

                    coef = values['coef']
                    x0 = values['x0']
                    alpha = values['alpha']
                    strength = values['strength']
                    length = values['length']
                    weight = values['weight']

                    temp_data = data.copy().fillna(0)

                    temp_data[f"{marketing_var}_c"] = Carryover(strength=values['strength'],
                                                                length=int(values['length'])).fit(
                        np.array(temp_data[marketing_var]).reshape(-1, 1)).transform(
                        np.array(temp_data[marketing_var]).reshape(-1, 1))

                    temp_data[f"{marketing_var}_trans"] = Saturation(x0=values['x0'], alpha=values['alpha']).fit(
                        np.array(temp_data[f"{marketing_var}_c"]).reshape(-1, 1)).transform(
                        np.array(temp_data[f"{marketing_var}_c"]).reshape(-1, 1))

                    temp_debug_df[f"{marketing_var}_trans"] = temp_data[f"{marketing_var}_trans"] * values['coef'] * \
                                                              values['weight']

                    temp_debug_df['x0'] = x0
                    temp_debug_df['alpha'] = alpha
                    temp_debug_df['strength'] = strength
                    temp_debug_df['length'] = length
                    temp_debug_df['weight'] = weight
                    temp_debug_df['coef'] = coef

                    res_debug_df = res_debug_df.append(temp_debug_df.set_index(['Model', 'Index']))

                    del temp_debug_df

                res_debug_df.reset_index().to_excel('data/interim/step3/' + f"decomp_{i.split('_w_ROI_w_Impact.xlsx')[0]}.xlsx", index=False)

        # для каждого маркетиногового регрессора пробегаюсь в цикле по каждой модели, преобразовываю регрессор, домножаю на coef и weight
        # после чего склыдваю (изначальный датафрейм "marketing_regressors" состоит из 156 нулей)

        # marketing_regressors = pd.DataFrame(0, index=np.arange(len(data)), columns = [i + '_trans' for i in paid_vars_imp])
        marketing_regressors = pd.DataFrame(0, index=np.arange(len(data)),
                                            columns=[i + '_trans' for i in temp_tab.index])

        for i in (os.listdir('data/interim/step3/')):

            if 'ROI_w_Impact' in i and 'all' not in i:

                df = pd.read_excel('data/interim/step3/' + i)

                marketing_var = i.split('_w_ROI_w_Impact.xlsx')[0]

                for index, values in df.iterrows():
                    temp_data = data.copy().fillna(0)

                    temp_data[f"{marketing_var}_c"] = Carryover(strength=values['strength'],
                                                                length=int(values['length'])).fit(
                        np.array(temp_data[marketing_var]).reshape(-1, 1)).transform(
                        np.array(temp_data[marketing_var]).reshape(-1, 1))

                    temp_data[f"{marketing_var}_trans"] = Saturation(x0=values['x0'], alpha=values['alpha']).fit(
                        np.array(temp_data[f"{marketing_var}_c"]).reshape(-1, 1)).transform(
                        np.array(temp_data[f"{marketing_var}_c"]).reshape(-1, 1))

                    temp_data[f"{marketing_var}_trans"] = temp_data[f"{marketing_var}_trans"] * values['coef'] * values[
                        'weight']

                    marketing_regressors[f"{marketing_var}_trans"] += temp_data[f"{marketing_var}_trans"]

        # marketing_regressors.columns = [i.replace('_trans', '') for i in marketing_regressors.columns]

        # склеиваю два дата фрейма (с контекстыми переменными и маркетинговыми)
        final_transformed = pd.concat([context_regressors, marketing_regressors], axis=1)

        final_transformed_model = final_transformed.copy()
        final_transformed_model['sales'] = data['sales'].values

        # получаю файл для декомпозиции по правилам которые мы обговорили ранее
        base = latest_weights['Intercept'].values[0]  # intersept

        for i in [i for i in final_transformed.columns]:
            if (final_transformed[i] > 0).all():
                base += np.min(final_transformed[final_transformed[i] > 0][i])
                final_transformed[i] = final_transformed[i] - np.min(final_transformed[final_transformed[i] > 0][i])

            elif i == 'comps_SOM':
                base += np.max(final_transformed[final_transformed.index <= 62]['comps_SOM'])
                final_transformed['comps_SOM'] = final_transformed['comps_SOM'] - np.max(
                    final_transformed[final_transformed.index <= 62]['comps_SOM'])


            elif (final_transformed[i] < 0).all():
                base += np.max(final_transformed[final_transformed[i] < 0][i])
                final_transformed[i] = final_transformed[i] - np.max(final_transformed[final_transformed[i] < 0][i])

        final_transformed = final_transformed.assign(Base=base)
        last_roi = pd.read_excel('data/interim/step3/' + 'roi_11.xlsx')

        final_model = sm.ols(formula(final_transformed_model.drop('sales', axis=1)), data=final_transformed_model).fit(
            method="pinv")

        final_transformed_2 = final_transformed.copy()

        final_transformed_2['date'] = data['date']

        final_transformed_2['price'] = final_transformed_2['average_price_dish_region_smooth_5'] + final_transformed_2[
            'price_lag_new_smooth_40'] + final_transformed_2['dish_qnt_reg_negative']

        final_transformed_2['price_positive'] = np.where(final_transformed_2['price'] >= 0,
                                                         final_transformed_2['price'], 0)

        final_transformed_2['price_negative'] = np.where(final_transformed_2['price'] < 0, final_transformed_2['price'],
                                                         0)

        final_transformed_2['Base'] = final_transformed_2['Base'] + final_transformed_2['stores'] + final_transformed_2[
            'dummy_apr'] + final_transformed_2['back_to_school']

        final_transformed_2['competitors'] = final_transformed_2['test_mac_off'] + final_transformed_2['McD_leave'] + \
                                             final_transformed_2['ViT_2'] + final_transformed_2['competitors_list_tv'] + \
                                             final_transformed_2['big_hit']

        final_transformed_2['competitors_positive'] = np.where(final_transformed_2['competitors'] >= 0,
                                                               final_transformed_2['competitors'], 0)

        final_transformed_2['competitors_negative'] = np.where(final_transformed_2['competitors'] < 0,
                                                               final_transformed_2['competitors'], 0)

        # final_transformed_2['competitors'] =  final_transformed_2['competitors_list_tv'] + final_transformed_2['big_hit'] + final_transformed_2['comps_SOM']

        final_transformed_2['digital'] = final_transformed_2['digital_2020_2022Q1_imp_trans'] + final_transformed_2[
            'digital_none_youtube_imp_trans']

        final_transformed_2['covid'] = final_transformed_2['lockdown'] + final_transformed_2['new_covid'] + \
                                       final_transformed_2['covid_dummy']

        final_transformed_2['covid'] = np.where(final_transformed_2['covid'] >= 0, 0, final_transformed_2['covid'])

        # final_transformed_2['mc_leave'] = final_transformed_2['McD_leave']   #'covid_dummy', 'back_to_school'\

        final_transformed_2 = final_transformed_2.drop(['lockdown', 'new_covid', 'covid_dummy',
                                                        'ViT_2', 'big_hit', 'test_mac_off', \
                                                        'competitors', 'price', 'McD_leave', \
                                                        'digital_none_youtube_imp_trans', 'dish_qnt_reg_negative', \
                                                        'average_price_dish_region_smooth_5', 'price_lag_new_smooth_40', \
                                                        'dummy_apr', 'back_to_school', \
                                                        'digital_2020_2022Q1_imp_trans', 'stores', \
                                                        'competitors_list_tv'], axis=1)

        final_transformed_2.to_excel('data/interim/step3/' + 'decomp_debug_final_transformed_2.xlsx')

        final_transformed_calc = final_transformed.copy()

        final_transformed_calc.columns = [i.replace('_trans', '') for i in final_transformed.columns]

        final_transformed_2_calc = final_transformed_2.drop('date', axis=1).copy()

        final_transformed_2_calc = final_transformed_2_calc.loc[:, (final_transformed_2_calc > 0).any(axis=0)]

        final_transformed_2_calc.columns = [i.replace('_trans', '') for i in final_transformed_2_calc.columns]

        global_impact = {}

        assert 'Base' in final_transformed_2_calc.columns

        for i in paid_vars_imp:
            if i in final_transformed_calc.columns:
                global_impact[i] = (final_transformed_calc[i].sum() / final_transformed_2_calc.sum().sum()) * 100

        global_impact = pd.DataFrame.from_dict(global_impact, orient='index')
        global_impact.to_excel('data/interim/step3/' + 'global_impact.xlsx')

        final_transformed_3 = final_transformed.copy()

        final_transformed_3['Fitted_Sales_Decomp_Sum'] = final_transformed_2.sum(axis=1).values

        final_transformed_3['final_model_fittedvalues'] = final_model.fittedvalues

        final_transformed_3.to_excel('data/interim/step3/' + 'final_transformed_attribution_decomposition.xlsx', index=False)

        final_transformed_3.to_excel('data/interim/step3/' + 'final_transformed_attribution.xlsx', index=False)

        import plotly.graph_objects as go

        import plotly.offline as pyo

        # colors = [
        #     '#FF0000',  # Red
        #     '#FF6A00',  # Orange
        #     '#FFD800',  # Yellow
        #     '#B6FF00',  # Lime
        #     '#4CFF00',  # Bright Green
        #     '#00FF21',  # Green
        #     '#00FFAA',  # Aquamarine
        #     '#00FFFF',  # Cyan
        #     '#00AFFF',  # Sky Blue
        #     '#0026FF',  # Blue
        #     '#4800FF',  # Indigo
        #     '#8F00FF',  # Violet
        #     '#D400FF',  # Purple
        #     '#FF00DC',  # Magenta
        #     '#FF0066',  # Deep Pink
        #     '#404040',  # Dark Gray
        #     '#9F9F9F',  # Gray
        #     '#FFFFFF',  # White
        #     '#000000'   # Black
        # ]

        colors = [
            '#FF0000',  # Red
            '#FF6A00',  # Orange
            '#FFD800',  # Yellow
            '#B6FF00',  # Lime
            '#4CFF00',  # Bright Green
            '#00FF21',  # Green
            '#00FFAA',  # Aquamarine
            '#00FFFF',  # Cyan
            '#00AFFF',  # Sky Blue
            '#0026FF',  # Blue
            '#4800FF',  # Indigo
            '#8F00FF',  # Violet
            '#D400FF',  # Purple
            '#FF00DC',  # Magenta
            '#FF0066',  # Deep Pink
            '#FF5050',  # Light Red
            '#339966',  # Sea Green
            '#9933FF',  # Amethyst
            '#FF99CC'  # Pink
        ]

        # Creating area plots
        final_transformed_2.set_index('date', inplace=True)
        fig = go.Figure()
        for i, col in enumerate(final_transformed_2.columns):
            stack_group = 'positive' if final_transformed_2[col].mean() >= 0 else 'negative'
            fig.add_trace(go.Scatter(x=final_transformed_2.index, y=final_transformed_2[col],
                                     mode='lines', stackgroup=stack_group,
                                     line=dict(width=0.5, color=colors[i % len(colors)]),
                                     # hoveron='fills',
                                     hovertemplate='Date: %{x}<br>' + col + ': %{y}<extra></extra>', name=col))

        # Adding the line plots
        fig.add_trace(go.Scatter(x=final_transformed_2.index, y=data['sales'], mode='lines', name='Real Sales',
                                 line=dict(color='black', dash='dash', width=2),
                                 hovertemplate='Date: %{x}<br>Real Checks: %{y}<extra></extra>'))

        # fig.add_trace(go.Scatter(x=final_transformed_2.index, y=final_transformed_2.sum(axis=1),
        #                          mode='lines', name='Fitted Sales Decomp Sum',
        #                          line=dict(color='blue', width=3),
        #                          hovertemplate='Date: %{x}<br>Fitted Sales Decomp Sum: %{y}<extra></extra>'))

        fig.add_trace(go.Scatter(x=final_transformed_2.index, y=final_model.fittedvalues,
                                 mode='lines', name='Fitted Sales Best Model',
                                 line=dict(color='red', width=3),
                                 hovertemplate='Date: %{x}<br>Fitted Sales Best Model: %{y}<extra></extra>'))

        # Example: Adding additional line trace
        # Assuming you have additional data in some variables called additional_x and additional_y
        # additional_x = [...]
        # additional_y = [...]
        # fig.add_trace(go.Scatter(x=additional_x, y=additional_y, mode='lines', name='Additional Line',
        #                          line=dict(color='green', width=2),
        #                          hovertemplate='Date: %{x}<br>Value: %{y}<extra></extra>'))

        # title='Predicted Sales and Breakdown',

        # Setting title and axes labels
        fig.update_layout(xaxis_title='Date', yaxis_title='Sales',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                          legend_title="Channels")

        # Saving the figure as an interactive HTML file
        pyo.plot(fig, filename='Decomposition.html')

        final_transformed_model['Base'] = final_transformed_2['Base']

        final_transformed_model['date'] = pd.to_datetime(data['date'])

        quarterly_data_impact = pd.DataFrame(
            final_transformed_model.groupby(final_transformed_model['date'].dt.to_period('Q')).sum()).reset_index()
        yearly_data_impact = pd.DataFrame(
            final_transformed_model.groupby(final_transformed_model['date'].dt.to_period('Y')).sum()).reset_index()

        columns_impact = [i for i in \
                          list(context_regressors.columns) + list(marketing_regressors.columns)]

        quarterly_data_impact = quarterly_data_impact[columns_impact + ['Base']]
        quarterly_data_impact = quarterly_data_impact[quarterly_data_impact.columns[quarterly_data_impact.gt(0).all()]]

        quarterly_data_impact['sum'] = quarterly_data_impact.sum(axis=1)
        quarterly_data_impact['date'] = pd.DataFrame(
            final_transformed_model.groupby(final_transformed_model['date'].dt.to_period('Q')).sum()).reset_index()[
            'date']

        yearly_data_impact = yearly_data_impact[columns_impact + ['Base']]
        yearly_data_impact = yearly_data_impact[yearly_data_impact.columns[yearly_data_impact.gt(0).all()]]

        yearly_data_impact['sum'] = yearly_data_impact.sum(axis=1)
        yearly_data_impact['date'] = pd.DataFrame(
            final_transformed_model.groupby(final_transformed_model['date'].dt.to_period('Y')).sum()).reset_index()[
            'date']

        temp_tab_new = temp_tab.copy()

        temp_tab_new['weighted_impact_2020'] = (temp_tab['weighted_impact_2020'] + 1e-10) / \
                                               yearly_data_impact.loc[yearly_data_impact['date'] == '2020'][
                                                   'sum'].values[0]

        temp_tab_new['weighted_impact_2021'] = (temp_tab['weighted_impact_2021'] + 1e-10) / \
                                               yearly_data_impact.loc[yearly_data_impact['date'] == '2021'][
                                                   'sum'].values[0]

        temp_tab_new['weighted_impact_2022_Q1'] = (temp_tab['weighted_impact_2022_Q1'] + 1e-10) / \
                                                  quarterly_data_impact.loc[quarterly_data_impact['date'] == '2022Q1'][
                                                      'sum'].values[0]

        temp_tab_new['weighted_impact_2022_Q2'] = (temp_tab['weighted_impact_2022_Q2'] + 1e-10) / \
                                                  quarterly_data_impact.loc[quarterly_data_impact['date'] == '2022Q2'][
                                                      'sum'].values[0]

        temp_tab_new['weighted_impact_2022_Q3'] = (temp_tab['weighted_impact_2022_Q3'] + 1e-10) / \
                                                  quarterly_data_impact.loc[quarterly_data_impact['date'] == '2022Q3'][
                                                      'sum'].values[0]

        temp_tab_new['weighted_impact_2022_Q4'] = (temp_tab['weighted_impact_2022_Q4'] + 1e-10) / \
                                                  quarterly_data_impact.loc[quarterly_data_impact['date'] == '2022Q4'][
                                                      'sum'].values[0]

        temp_tab_new['weighted_impact_2023_Q1'] = (temp_tab['weighted_impact_2023_Q1'] + 1e-10) / \
                                                  quarterly_data_impact.loc[quarterly_data_impact['date'] == '2023Q1'][
                                                      'sum'].values[0]

        temp_tab_new[[i for i in temp_tab_new.columns if 'impact' in i]].to_excel('data/interim/step3/' + 'impact%.xlsx')

        temp_tab = pd.read_excel('data/interim/step3/' + 'roi_breakdown.xlsx')
        weighted_roi = temp_tab.copy()

        temp_tab_new[['weighted_roi_2020', 'weighted_roi_2021', 'weighted_roi_2022_Q1', 'weighted_roi_2022_Q2',
                      'weighted_roi_2022_Q3', 'weighted_roi_2022_Q4', \
                      'weighted_impact_2020', 'weighted_impact_2021', 'weighted_impact_2022_Q1',
                      'weighted_impact_2022_Q2', 'weighted_impact_2022_Q3', 'weighted_impact_2022_Q4']] \
            .to_excel('data/interim/step3/' + 'ROI_IMPACT%.xlsx')

        df = data.copy()

        df['avg_check'] = data['sales'] / data['checks']

        df = df.rename(columns={'nat_tv_vfm_spend': 'nat_tv_wo2020_vfm_spend_sov'})

        df = df.rename(columns={'nat_tv_product_spend': 'nat_tv_wo2020_product_spend_sov'})

        df = df.rename(columns={'nat_tv_angus_spend': 'nat_tv_wo2020_angus_spend_norm_sov'})

        df['date'] = pd.to_datetime(df['date'])

        quarterly_data = pd.DataFrame(df.groupby(df['date'].dt.to_period('Q'))['avg_check'].mean()).reset_index()

        for i in weighted_roi['index']:

            if 'imp' in i:
                temp_df = pd.DataFrame(
                    df.groupby(df['date'].dt.to_period('Q'))[i, i.replace('imp', 'spend')].sum()).reset_index()
                temp_df[i + '_CPP'] = temp_df[i.replace('imp', 'spend')] / temp_df[i]

                quarterly_data = pd.concat([quarterly_data, temp_df.drop('date', axis=1).fillna(0)], axis=1)

                del temp_df

        quarterly_data = quarterly_data[quarterly_data['date'] >= '2022Q1']

        df = data.copy()

        df['avg_check'] = data['sales'] / data['checks']

        df = df.rename(columns={'nat_tv_vfm_spend': 'nat_tv_wo2020_vfm_spend_sov'})

        df = df.rename(columns={'nat_tv_product_spend': 'nat_tv_wo2020_product_spend_sov'})

        df = df.rename(columns={'nat_tv_angus_spend': 'nat_tv_wo2020_angus_spend_norm_sov'})

        df['date'] = pd.to_datetime(df['date'])

        yearly_data = pd.DataFrame(df.groupby(df['date'].dt.to_period('Y'))['avg_check'].mean()).reset_index()

        quarterly_data = pd.DataFrame(df.groupby(df['date'].dt.to_period('Q'))['avg_check'].mean()).reset_index()

        for i in weighted_roi['index']:

            if 'imp' in i:
                temp_df = pd.DataFrame(
                    df.groupby(df['date'].dt.to_period('Y'))[i, i.replace('imp', 'spend')].sum()).reset_index()
                temp_df[i + '_CPP'] = temp_df[i.replace('imp', 'spend')] / temp_df[i]

                yearly_data = pd.concat([yearly_data, temp_df.drop('date', axis=1).fillna(0)], axis=1)

                del temp_df

        roi_decomp = {}

        for i in weighted_roi['index'].unique():

            if 'imp' not in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2021'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2021'].values[0] \
                                   * ((yearly_data.loc[yearly_data['date'] == '2021']['avg_check'].values[0] \
                                       / yearly_data.loc[yearly_data['date'] == '2020']['avg_check'].values[0]) - 1)

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2021'].values[0] \
                             - avg_check_change \
                             - weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2020'].values[0]

                roi_decomp[(i, '2021')] = [avg_check_change, effeciency]

            elif 'imp' in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2021'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2021'].values[0] \
                                   * ((yearly_data.loc[yearly_data['date'] == '2021']['avg_check'].values[0] \
                                       / yearly_data.loc[yearly_data['date'] == '2020']['avg_check'].values[0]) - 1)

                CPP_change = -1 * (weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2021'].values[0] \
                                   * ((yearly_data.loc[(yearly_data['date'] == '2021')][i + '_CPP'].values[0] \
                                       / yearly_data.loc[(yearly_data['date'] == '2020')][i + '_CPP'].values[0]) - 1))

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2021'].values[0] \
                             - avg_check_change - \
                             weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2020'].values[0] - CPP_change
                roi_decomp[(i, '2021')] = [avg_check_change, effeciency, CPP_change]

        df = data.copy()

        df['avg_check'] = data['sales'] / data['checks']

        df = df.rename(columns={'nat_tv_vfm_spend': 'nat_tv_wo2020_vfm_spend_sov'})

        df = df.rename(columns={'nat_tv_product_spend': 'nat_tv_wo2020_product_spend_sov'})

        df = df.rename(columns={'nat_tv_angus_spend': 'nat_tv_wo2020_angus_spend_norm_sov'})

        df['date'] = pd.to_datetime(df['date'])

        quarterly_data = pd.DataFrame(df.groupby(df['date'].dt.to_period('Q'))['avg_check'].mean()).reset_index()

        for i in weighted_roi['index']:

            if 'imp' in i:
                temp_df = pd.DataFrame(
                    df.groupby(df['date'].dt.to_period('Q'))[i, i.replace('imp', 'spend')].sum()).reset_index()
                temp_df[i + '_CPP'] = temp_df[i.replace('imp', 'spend')] / temp_df[i]

                quarterly_data = pd.concat([quarterly_data, temp_df.drop('date', axis=1).fillna(0)], axis=1)

                del temp_df

        quarterly_data = quarterly_data[quarterly_data['date'] >= '2022Q1']

        for i in weighted_roi['index'].unique():

            if 'imp' not in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2022Q1']['avg_check'].values[0] \
                                       / yearly_data.loc[yearly_data['date'] == '2021']['avg_check'].values[0]) - 1)

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[0] \
                             - avg_check_change \
                             - weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2021'].values[0]

                roi_decomp[(i, '2022Q1')] = [avg_check_change, effeciency]

            elif 'imp' in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2022Q1']['avg_check'].values[0] \
                                       / yearly_data.loc[yearly_data['date'] == '2021']['avg_check'].values[0]) - 1)

                CPP_change = -1 * (weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[0] \
                                   * ((quarterly_data.loc[(quarterly_data['date'] == '2022Q1')][i + '_CPP'].values[0] \
                                       / yearly_data.loc[(yearly_data['date'] == '2021')][i + '_CPP'].values[0]) - 1))

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[0] \
                             - avg_check_change - \
                             weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2021'].values[0] - CPP_change

                roi_decomp[(i, '2022Q1')] = [avg_check_change, effeciency, CPP_change]

        for i in weighted_roi['index'].unique():

            if 'imp' not in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q2'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q2'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2022Q2']['avg_check'].values[0] \
                                       / quarterly_data.loc[quarterly_data['date'] == '2022Q1']['avg_check'].values[
                                           0]) - 1)

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q2'].values[0] \
                             - avg_check_change \
                             - weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[0]

                roi_decomp[(i, '2022Q2')] = [avg_check_change, effeciency]

            elif 'imp' in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q2'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q2'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2022Q2']['avg_check'].values[0] \
                                       / quarterly_data.loc[quarterly_data['date'] == '2022Q1']['avg_check'].values[
                                           0]) - 1)

                CPP_change = -1 * (weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[0] \
                                   * ((quarterly_data.loc[(quarterly_data['date'] == '2022Q2')][i + '_CPP'].values[0] \
                                       / quarterly_data.loc[(quarterly_data['date'] == '2022Q1')][i + '_CPP'].values[
                                           0]) - 1))

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q2'].values[0] \
                             - avg_check_change - \
                             weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q1'].values[
                                 0] - CPP_change

                roi_decomp[(i, '2022Q2')] = [avg_check_change, effeciency, CPP_change]

        for i in weighted_roi['index'].unique():

            if 'imp' not in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2022Q3']['avg_check'].values[0] \
                                       / quarterly_data.loc[quarterly_data['date'] == '2022Q2']['avg_check'].values[
                                           0]) - 1)

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[0] \
                             - avg_check_change \
                             - weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q2'].values[0]

                roi_decomp[(i, '2022Q3')] = [avg_check_change, effeciency]

            elif 'imp' in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2022Q3']['avg_check'].values[0] \
                                       / quarterly_data.loc[quarterly_data['date'] == '2022Q2']['avg_check'].values[
                                           0]) - 1)

                CPP_change = -1 * (weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q2'].values[0] \
                                   * ((quarterly_data.loc[(quarterly_data['date'] == '2022Q3')][i + '_CPP'].values[0] \
                                       / quarterly_data.loc[(quarterly_data['date'] == '2022Q2')][i + '_CPP'].values[
                                           0]) - 1))

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[0] \
                             - avg_check_change - \
                             weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q2'].values[
                                 0] - CPP_change

                roi_decomp[(i, '2022Q3')] = [avg_check_change, effeciency, CPP_change]

        for i in weighted_roi['index'].unique():

            if 'imp' not in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q4'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q4'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2022Q4']['avg_check'].values[0] \
                                       / quarterly_data.loc[quarterly_data['date'] == '2022Q3']['avg_check'].values[
                                           0]) - 1)

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q4'].values[0] \
                             - avg_check_change \
                             - weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[0]

                roi_decomp[(i, '2022Q4')] = [avg_check_change, effeciency]

            elif 'imp' in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q4'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q4'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2022Q4']['avg_check'].values[0] \
                                       / quarterly_data.loc[quarterly_data['date'] == '2022Q3']['avg_check'].values[
                                           0]) - 1)

                CPP_change = -1 * (weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[0] \
                                   * ((quarterly_data.loc[(quarterly_data['date'] == '2022Q4')][i + '_CPP'].values[0] \
                                       / quarterly_data.loc[(quarterly_data['date'] == '2022Q3')][i + '_CPP'].values[
                                           0]) - 1))

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q4'].values[0] \
                             - avg_check_change - \
                             weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[
                                 0] - CPP_change

                roi_decomp[(i, '2022Q4')] = [avg_check_change, effeciency, CPP_change]

        for i in weighted_roi['index'].unique():

            if 'imp' not in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2023_Q1'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2023_Q1'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2023Q1']['avg_check'].values[0] \
                                       / quarterly_data.loc[quarterly_data['date'] == '2022Q4']['avg_check'].values[
                                           0]) - 1)

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2023_Q1'].values[0] \
                             - avg_check_change \
                             - weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q4'].values[0]

                roi_decomp[(i, '2023Q1')] = [avg_check_change, effeciency]

            elif 'imp' in i and weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2023_Q1'].values[0] > 0:

                avg_check_change = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2023_Q1'].values[0] \
                                   * ((quarterly_data.loc[quarterly_data['date'] == '2023Q1']['avg_check'].values[0] \
                                       / quarterly_data.loc[quarterly_data['date'] == '2022Q4']['avg_check'].values[
                                           0]) - 1)

                CPP_change = -1 * (weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q3'].values[0] \
                                   * ((quarterly_data.loc[(quarterly_data['date'] == '2023Q1')][i + '_CPP'].values[0] \
                                       / quarterly_data.loc[(quarterly_data['date'] == '2022Q4')][i + '_CPP'].values[
                                           0]) - 1))

                effeciency = weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2023_Q1'].values[0] \
                             - avg_check_change - \
                             weighted_roi.loc[(weighted_roi['index'] == i)]['weighted_roi_2022_Q4'].values[
                                 0] - CPP_change

                roi_decomp[(i, '2023Q1')] = [avg_check_change, effeciency, CPP_change]

        max_length = max(len(v) for v in roi_decomp.values())

        roi_decomp_ans = pd.DataFrame({k: v + [np.nan] * (max_length - len(v)) for k, v in roi_decomp.items()})
        roi_decomp_ans.index = ['avg_check_change', 'effeciency', 'CPP_change']
        roi_decomp_ans.to_excel('data/interim/step3/' + 'roi_decomp.xlsx')

        return self