import inline
import pandas as pd
import numpy as np
import pickle   # сохранение модели

import matplotlib
#import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#%matplotlib inline

# 2. Разделение датасета
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

# 3. Модели
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# 4. Метрики качества
from sklearn.metrics import mean_squared_error as mse, r2_score as r2

# 5. Для визуализации внешних картинок в ноутбуке
from IPython.display import Image


import matplotlib.image as img
from scipy.stats import mode
import datetime
#%matplotlib inline

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

def task1():
    x = 1.6453
    y = np.power(np.sin(np.pi / 6 - 1), 2)
    y += np.power(3 + np.power(x, 2), 0.25)
    y -= np.power(np.log10(np.power(x, 3) - 1), 3)
    temp = np.arcsin(x / 2) - 1.756 * np.power(1 / 10, 2)
    y /= temp
    print("Результат:", y)


def task2():
    variant = 23
    n = 12
    m = 3
    x = np.ones((n, m))
    x[:, 1] = np.random.randint(variant, variant + 12, n)
    x[:, 2] = np.random.randint(60, 82, n)
    print("Матрица X:\n", x)
    y = np.random.uniform(13.5, 18.6, [12, 1])
    print("Вектор-столбец Y:\n", y)
    x_t = x.transpose()
    print("Транспонированная матрица X_t:\n", x_t)
    c = np.linalg.inv(x_t.dot(x))
    print("Обратная матрица C произведения X_t на X:\n", c)
    d = x_t.dot(y)
    print("Матрица D, равная произведению матрицы X_t на Y:\n", d)
    a = c.dot(d)
    print("Вектор оценок A, равный произведению C и D:\n", a)
    y_temp = np.array([[0] * 1 for i in range(n)], dtype=float)
    for i in range(0, n):
        y_temp[i][0] = a[0, 0] + a[1, 0] * x[i, 1] + a[2, 0] * x[i, 2]
    print("Проверка:\n", y_temp)


def task3():
    df = pd.read_csv("test.csv")
    print(df.shape)
    print(df.dtypes)
    # отбор числовых колонок
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values
    print(numeric_cols)

    # отбор нечисловых колонок
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    non_numeric_cols = df_non_numeric.columns.values
    print(non_numeric_cols)

    cols = df.columns[:19]  # первые 30 колонок
    # определяем цвета
    # желтый - пропущенные данные, синий - не пропущенные
    colours = ['#6900C6', '#ff0000']
    sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))

    for col in df.columns:
        missing = df[col].isnull()
        num_missing = np.sum(missing)

        if num_missing > 0:
            print('created missing indicator for: {}'.format(col))
            df['{}_ismissing'.format(col)] = missing

    # затем на основе индикатора строим гистограмму
    ismissing_cols = [col for col in df.columns if 'ismissing' in col]
    df['num_missing'] = df[ismissing_cols].sum(axis=1)

    df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')

    print("Процент пропусков:")
    for col in df.columns:
        pct_missing = np.mean(df[col].isnull())
        print('{} - {}%'.format(col, round(pct_missing * 100)))
    df["LifeSquare"].hist(bins=100)

    # посмотрим описательные статистики
    # аномальные минимальные и максимальные значения у признаков Rooms, Square, LifeSquare, KitchenSquare
    # также видны пропуски для признаков LifeSquare и Healthcare_1
    df.describe()


def task4():
    return


def menu():
    while True:
        print("Список заданий:\n "
              "1. Вычисление выражения\n "
              "2. Нахождение оценки уровня регрессии\n "
              "3. Графики\n "
              "4. Pandas\n "
              "0. Выход")
        variant = input("Выберите задание: ")
        try:
            variant = int(variant)
        except ValueError:
            print("Введите целочисленное число!")
            continue
        if variant > 4 or variant < 0:
            print("Ошибка, введите число в заданном интервале!")
        else:
            match variant:
                case 1:
                    task1()
                case 2:
                    task2()
                case 3:
                    task3()
                case 4:
                    task4()
                case 0:
                    break
                case _:
                    print("Ошибка!")
                    return -1
    return 0


if __name__ == '__main__':
    menu()
