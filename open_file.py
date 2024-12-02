import pandas as pd
import matplotlib.pyplot as plt


def show_data(theta_0, theta_1, df):
    plt.scatter(df['km'], df['price'])
    plt.plot(df['km'], theta_0 + theta_1 * df['km'])
    plt.title("Test")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.show()


# def mean_squared_error(theta_0, theta_1, df):
    # MSE = 0
    # lenght = len(df)
    # for i in range(lenght):
        # x = df.iloc[i].km
        # y = df.iloc[i].price
        # MSE += 1/lenght * (y - (theta_0 + theta_1 * x)) ** 2  

def normalized_data(df) -> pd.DataFrame:
    min_km = df['km'].min()
    max_km = df['km'].max()
    min_price = df['price'].min()
    max_price = df['price'].max()

    df['km_norm'] = (df['km'] - min_km) / (max_km - min_km)
    df['price_norm'] = (df['price'] - min_price) / (max_price - min_price)
    print(df)


def denormilize_value(theta_0, theta_1, df):
    min_km = df['km'].min()
    max_km = df['km'].max()
    min_price = df['price'].min()
    max_price = df['price'].max()

    scale_price = max_price - min_price
    scale_km = max_km - min_km

    theta_1 = theta_1 * scale_price / scale_km
    theta_0 = theta_0 * scale_price + min_price - theta_1 * min_km
    return theta_0, theta_1


def estimate_price(theta_0, theta_1, x):
    return theta_0 + theta_1 * x


def gradient_descent(theta_0, theta_1, df, learningRate):
    m = len(df)
    tmp_0 = 0
    tmp_1 = 0
    for i in range(m):
        x = df.iloc[i].km_norm
        y = df.iloc[i].price_norm
        y_hat = estimate_price(theta_0, theta_1 , x)

        tmp_0 += y_hat - y
        tmp_1 += (y_hat - y) * x
    tmp_0 = theta_0 - (learningRate * (1/m) * tmp_0)
    tmp_1 = theta_1 - (learningRate * (1/m) * tmp_1)
    return tmp_0, tmp_1



# def gradient_descent(theta_0, theta_1  , df, learningRate):
#     m = len(df)
#     tmp_0 = 0
#     tmp_1 = 0
#     error = 0
#     for i in range(m):
#         x = df.iloc[i].x_norm
#         y = df.iloc[i].price
#         y_chapeau = estimate_price(theta_0, theta_1, x)
#         error += (y - y_chapeau) ** 2
#         #print(f"X[{i}] : {x}, Y : {y}, y_hat : {y_chapeau}")
#         #tmp_0 += y_chapeau - y
#         #tmp_1 += (y_chapeau - y)
#         tmp_0 += y_chapeau - y
#         tmp_1 += (y_chapeau - y) * x 
#         #print(f"[{i}]   {tmp_0}, {tmp_1}")

#     tmp_0 /= m
#     tmp_1 /= m
#     #print(f'{error=}')
#     new_0 = theta_0 - tmp_0 * learningRate
#     new_1 = theta_1 - tmp_1 * learningRate
#     return new_0, new_1

# def gradient_descent(theta_0, theta_1, df, learningRate):
#     m = len(df)
#     tmp_0 = 0
#     tmp_1 = 0

#     for i in range(m):
#         x = df.iloc[i].km / df['km'].max()
#         y = df.iloc[i].price / df['price'].max()
#         error = y - (theta_0 + theta_1 * x)

#         tmp_0 += error
#         tmp_1 = error * x

#     tmp_0 = (1 / m) * tmp_0
#     tmp_1 = (1 / m) * tmp_1

#     theta_0 -= learningRate * tmp_0
#     theta_1 -= learningRate * tmp_1

#     print(f"Updated Parameters: theta_0={theta_0}, theta_1={theta_1}")
#     return theta_0, theta_


def main():
    df = pd.read_csv("data.csv")
    theta_0 = 0
    theta_1 = 0
    learningRate = 0.1
    epochs = 1000

    normalized_data(df)
    for i in range(epochs):
        if i % 100 == 0:
            print(f"epochs = {i}")
        theta_0, theta_1 = gradient_descent(theta_0, theta_1, df, learningRate)
        print(f"[{i}]  {theta_0}  |  {theta_1}")

    theta_0, theta_1 = denormilize_value(theta_0, theta_1, df)
    show_data(theta_0, theta_1, df)


if __name__ == "__main__":
    main()
