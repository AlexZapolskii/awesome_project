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
from utils.utils import Step2


if __name__ == '__main__':
    PATH = "data/raw/total_df.csv"
    current_region = 'юг'

    df = load_transform_dataset(PATH, current_region)

    optimize(df) # запускает оптимизатор и сохраняет результаты в папку interim

    # Шаг 2 - расчет ROI
    Step2(current_region, df).process_files().fit().ROI()
