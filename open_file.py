import pandas as pd
import matplotlib.pyplot as plt


def show_data():
    df = pd.read_csv("data.csv")
    #for x, y in zip(df['km'][x], df['price'][y])
    #tmp1 = 0.001 * (1/len(df)) for 
    #tmp2 = 
    plt.scatter(df['km'], df['price'])
    plt.title("Test")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.show()


def main():
    show_data()


if __name__ == "__main__":
    main()