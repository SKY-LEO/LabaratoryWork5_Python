import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


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
    dataset = pd.read_csv("test.csv")
    df = dataset.sample(n=1000)
    # посмотрим описательные статистики
    # аномальные минимальные и максимальные значения у признаков Rooms, Square, LifeSquare, KitchenSquare
    # также видны пропуски для признаков LifeSquare и Healthcare_1
    print("Общие сведения:\n", df.describe().to_string())
    print("Размерность датасета (строки, столбцы):", df.shape)
    print("Типы данных столбцов (название, тип):\n", df.dtypes, sep="")

    print("\nРабота с пропусками значений.")
    cols = df.columns[:19]
    colours = ["#6900C6", "#ff0000"]
    sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
    plt.title("Тепловая карта")
    plt.show()

    print("\nПроцентный список пропущенных данных:")
    for column in df.columns:
        percent_missing = np.mean(df[column].isnull())
        print("{} - {}%".format(column, round(percent_missing * 100)))
    digital_features = df.select_dtypes(exclude=["object"])
    digital_features.hist(figsize=(16, 12), bins=30)
    plt.title("Данные в графическом представлении")
    plt.show()

    # рисунок 3, ящик с усами
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df[["Square", "LifeSquare", "KitchenSquare"]], orient="h")
    plt.xscale("symlog")
    plt.xlim(left=-1)
    plt.title("Ящик с усами")
    plt.show()

    # гистограмма
    sns.histplot(data=df["Square"])
    plt.title("Гистограмма Square")
    plt.show()

    sns.histplot(data=df["LifeSquare"])
    plt.title("Гистограмма LifeSquare")
    plt.show()

    sns.histplot(data=df["KitchenSquare"])
    plt.title("Гистограмма KitchenSquare")
    plt.show()

    # заполнение пропусков
    df = df.sort_values(by="Square")
    df.LifeSquare.fillna(method="ffill", inplace=True)
    df = df.sort_values(by="DistrictId")
    df.Healthcare_1.fillna(method="ffill", inplace=True)
    df = df.sort_values(by="Id")

    # заменим аномальные значения количества комнат на медианы
    print("\nЗаменим аномальные значения количества комнат на медианы:")
    df.loc[df["Rooms"].isin([0, 10, 17, 19]), "Rooms"] = int(df["Rooms"].median())
    print(df.describe().to_string())

    # заменим аномальные значения площадей на медианы
    print("\nЗаменим аномальные значения площадей на медианы:")
    df.loc[(df["LifeSquare"] > 80) | (df["LifeSquare"] < 5), "LifeSquare"] = df["LifeSquare"].median()
    df.loc[(df["Square"] > 100) | (df["Square"] < 10), "Square"] = df["Square"].median()
    df.loc[(df["KitchenSquare"] > 15) | (df["KitchenSquare"] < 1), "KitchenSquare"] = df["KitchenSquare"].median()
    print(df.describe().to_string())

    digital_features = df.select_dtypes(exclude=["object"])
    digital_features.hist(figsize=(16, 12), bins=30)
    plt.title("Данные в графическом представлении")
    plt.show()

    print("\nКоличество квартир по комнатам:")
    print(df["Rooms"].value_counts().rename_axis("Rooms").reset_index(name="Amount"))

    new_df = df[["DistrictId", "Rooms"]]
    new_df["Rooms1"] = 1

    table = pd.pivot_table(new_df, index="DistrictId", values="Rooms1", columns="Rooms", fill_value=0, aggfunc=np.sum)
    print(table.to_string())

    df.to_csv("surname.csv", index=False)


def task4():
    arr = []
    x = 3.567
    interval = (-5, 12)
    delta_a = 0.5

    arguments = np.arange(interval[0], interval[1], delta_a)
    for a in arguments:
        arr.append(np.power((1. / np.tan(x)), 3) + 2.24 * a * x)
    print(arr)
    print("Максимальный элемент списка:", np.max(arr))
    print("Минимальный элемент списка:", np.min(arr))
    print("Среднее значение списка:", np.mean(arr))
    print("Длина списка:", np.size(arr))

    sorted_array = np.sort(arr)
    print("Отсортированный по возрастанию список:", sorted_array)

    plt.plot(arguments, arr, "g", marker="o")
    plt.plot(interval, np.full(2, np.mean(arr)), "r")
    plt.ylabel("Значение")
    plt.xlabel("Аргумент")
    plt.show()

    interval_x = (0, 3)
    delta_x = 0.1
    interval_y = (3, 9)
    delta_y = 0.2

    arguments_x = np.arange(interval_x[0], interval_x[1], delta_x)
    arguments_y = np.arange(interval_y[0], interval_y[1], delta_y)
    val_1 = np.array([x ** 0.25 + y ** 0.25 for x, y in zip(arguments_x, arguments_y)])
    val_2 = np.array([x ** 2 - y ** 2 for x, y in zip(arguments_x, arguments_y)])
    val_3 = np.array([2 * x + 3 * y for x, y in zip(arguments_x, arguments_y)])
    val_4 = np.array([x ** 2 + y ** 2 for x, y in zip(arguments_x, arguments_y)])
    val_5 = np.array([2 + 2 * x + 2 * y - x ** 2 - y ** 2 for x, y in zip(arguments_x, arguments_y)])
    ax = plt.axes(projection="3d")
    ax.plot3D(arguments_x, arguments_y, val_1)
    ax.plot3D(arguments_x, arguments_y, val_2)
    ax.plot3D(arguments_x, arguments_y, val_3)
    ax.plot3D(arguments_x, arguments_y, val_4)
    ax.plot3D(arguments_x, arguments_y, val_5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("3D-график")
    plt.show()


def menu():
    while True:
        print("Список заданий:\n "
              "1. Вычисление выражения\n "
              "2. Нахождение оценки уровня регрессии\n "
              "3. Pandas\n "
              "4. Графики\n "
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
