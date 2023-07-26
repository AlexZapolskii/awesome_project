import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.utils import *
from utils.enums import *
def load_transform_dataset(path):
    """
    Загружает исходные данные и
    приводит в нужный формат
        """
    path = file
    df = pd.read_csv(path).fillna(0)
    df['date'] = pd.to_datetime(df['date'])
    df['mc_leave'] = np.where(df['date'] >= '2022-03-14', 1, 0)
    df['lockdown'] = np.where((df['date'] >= "2021-10-28") & (df['date'] <= "2021-11-07"), 1, 0)
    df['nat_tv_spend_2021_2022'] = np.where(df['date'].dt.year >= 2021, df['nat_tv_spend'], 0)

    df = df.set_index("date")
    df = df.loc[df.index >= "2020-01-01"]
    df = df[df['region'] == current_region]

    decomposition = seasonal_decompose(df[["sales"]], model="multiplicative", period= int(len(df)/2))
    df["trend"] = range(len(df))

    df['seasonality'] = df['Seas_calc_2016_2022']
    df['nat_tv_wo2020_product_imp_sov'] = df['nat_tv_wo2020_product_imp_norm']*0.5 + df['SOV_product_norm']*0.5
    df['nat_tv_wo2020_vfm_imp_sov'] = df['nat_tv_wo2020_vfm_imp_norm']*0.5 + df['SOV_vfm_norm']*0.5

    df["digital_2020_2022Q1_imp"] = df[digital_list].sum(axis=1)
    df["digital_2020_2022Q1_imp"] = df[df.index <= '2022-04-01']['digital_2020_2022Q1_imp']

    df["digital_2020_2022Q1_spend"] = df[digital_spend_list].sum(axis=1)
    df["digital_2020_2022Q1_spend"] = df[df.index <= '2022-04-01']['digital_2020_2022Q1_spend']

    df["digital_none_youtube_imp"] = df[digital_list].sum(axis=1)
    df["digital_none_youtube_imp"] = df[df.index > '2022-04-01']['digital_none_youtube_imp']

    df["digital_none_youtube_spend"] = df[digital_spend_list].sum(axis=1)
    df["digital_none_youtube_spend"] = df[df.index > '2022-04-01']['digital_none_youtube_spend']

    df = df.fillna(0)

    df['avg_check'] = 1
    df["competitors_list_tv"] = df[competitors_list_tv].sum(axis=1)

    return df