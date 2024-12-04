import pandas as pd


def normalized_data(df) -> pd.DataFrame:
    min_km = df['km'].min()
    max_km = df['km'].max()
    min_price = df['price'].min()
    max_price = df['price'].max()

    df['km_norm'] = (df['km'] - min_km) / (max_km - min_km)
    df['price_norm'] = (df['price'] - min_price) / (max_price - min_price)


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
        y_hat = estimate_price(theta_0, theta_1, x)

        tmp_0 += y_hat - y
        tmp_1 += (y_hat - y) * x
    tmp_0 = theta_0 - (learningRate * (1/m) * tmp_0)
    tmp_1 = theta_1 - (learningRate * (1/m) * tmp_1)
    return tmp_0, tmp_1


def main():
    try:
        df = pd.read_csv("data.csv")
        theta_0 = 0
        theta_1 = 0
        learningRate = 0.1
        epochs = 1000

        normalized_data(df)
        for i in range(epochs):
            theta_0, theta_1 = gradient_descent(theta_0, theta_1, df, learningRate)
        theta_0, theta_1 = denormilize_value(theta_0, theta_1, df)
        df_result = pd.DataFrame({'theta_0': [theta_0],
                                  'theta_1': [theta_1]})
        df_result.to_csv('model_bonus.csv')
        print("The training is finished. The result of training is in 'model_bonus.csv file.")
    except KeyboardInterrupt:
        print("The program was interrupted. Please try again")
    except Exception as e:
        print(f"{type(e).__name__}: Error")


if __name__ == "__main__":
    main()
