"""
модуль, который повторяет алгоритм первого ноутбучка
"""

import  pandas as pd
from matplotlib import pyplot as plt
pallete = plt.get_cmap('Set2')
import warnings

warnings.filterwarnings('ignore')
from data.load_data import load_transform_dataset
from models.optimizer import optimize
from utils.utils import Step2, Step4
from models.step3_draft import optimize_step3

if __name__ == '__main__':
    file = "data/raw/total_df.xlsx"
    df = pd.read_excel(file).fillna(0)
    df.to_csv('data/raw/total_df.csv', index=False)

    PATH = "data/raw/total_df.csv"
    current_region = 'юг'

    df, paid_vars_imp = load_transform_dataset(PATH, current_region)

    # df.to_csv('data/raw/final_df.csv')

    # optimize(df, paid_vars_imp) # запускает оптимизатор и сохраняет результаты в папку interim

    # Шаг 2 - расчет ROI
    # Step2(current_region, df).process_files().fit().ROI()

    # Шаг 3

    # optimize_step3(df, paid_vars_imp) # запускает оптимизатор и сохраняет результаты в папку interim

    print('Step 4 start')
    # Шаг 4
    Step4(df).process_files(df)
