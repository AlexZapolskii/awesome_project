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


if __name__ == '__main__':
    PATH = "data/raw/total_df.csv"
    current_region = 'юг'

    df = load_transform_dataset(PATH, current_region)

    optimize(df) # запускает оптимизатор и сохраняет результаты в папку interim

