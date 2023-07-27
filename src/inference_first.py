from matplotlib import pyplot as plt
pallete = plt.get_cmap('Set2')
import warnings

warnings.filterwarnings('ignore')
from src.utils.load_data import  load_transform_dataset
from src.models.optimizer import optimize


if __name__ == '__main__':
    PATH = "data/raw/total_df.csv"
    df = load_transform_dataset(PATH)

    # TODO: call optimizer
    #       В оптимизаторе перебираются параметры
    #       Сохраняется результат - ексели (в interim или processed)

    optimize(df) # запускает оптимизатор и сохраняет результаты в папку interim

