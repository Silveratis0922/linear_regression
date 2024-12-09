import pandas as pd
import matplotlib.pyplot as plt


def save_data(df, prediction, x):
    """Save low and max data for the scale of the graph"""
    min_km = min(df['km'].min(), x)
    max_km = max(df['km'].max(), x)
    min_price = min(df['price'].min(), prediction)
    max_price = max(df['price'].max(), prediction)

    return min_km, min_price, max_km, max_price


def display_data():
    """Display true data"""
    df = pd.read_csv("data.csv")
    plt.scatter(df['km'], df['price'])
    plt.title("Graph of the real data")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.show()
    return df


def display_linear_regression(theta_0, theta_1, df):
    """Display true data with linear regression line through true data"""
    line = theta_0 + theta_1 * df['km']

    plt.scatter(df['km'], df['price'], label="real data")
    plt.plot(df['km'], line, 'k', label="linear regression")
    plt.legend()
    plt.title("Real data with linear regression")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.show()


def display_estimate_price(theta_0, theta_1, prediction, x, df):
    """Display the mileage given on your linear regression line"""
    min_km, min_price, max_km, max_price = save_data(df, prediction, x)
    arg = {"markersize": 8, "markeredgewidth": 2, "label": "prediction"}

    line_min = theta_0 + theta_1 * 0
    line_max = theta_0 + theta_1 * 500000
    xab = [0, 500000]
    yab = [line_min, line_max]
    plt.scatter(df['km'], df['price'], label="real data")
    plt.plot(xab, yab, 'k', label="linear regression")
    plt.plot(x, prediction, 'rx', **arg)
    plt.xlim(min_km - 25000, max_km + 25000)
    plt.ylim(min_price - 1000, max_price + 1000)
    plt.legend()
    plt.title(f"Price prediction for {x} mileage")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.show()


def model_result():
    """Read the csv file of the training model"""
    df = pd.read_csv("model_bonus.csv")
    theta_0 = df.iloc[0]["theta_0"]
    theta_1 = df.iloc[0]["theta_1"]
    return theta_0, theta_1


def accuracy(theta_0, theta_1, prediction, df):
    """Function for the Coefficient of determination calculation
       print the accuracy of the algorythm"""
    sst = 0
    ssr = 0
    y_mean = df['price'].mean()

    for i in range(len(df)):
        sst += (df.iloc[i].price - y_mean) ** 2
        ssr += (df.iloc[i].price - (theta_0 + theta_1 * df.iloc[i].km)) ** 2
    r_square = 1 - ssr / sst
    print(f"The accuracy of the algorith is {r_square * 100:.2f}%.")


def main():
    try:
        print("Enter the mileage for prediction: ")
        theta_0 = 0
        theta_1 = 0
        x = int(input())
        if x < 0:
            raise AssertionError("Milage cant't be under zero.")
        df = display_data()
        theta_0, theta_1 = model_result()
        display_linear_regression(theta_0, theta_1, df)
        prediction = theta_0 + theta_1 * x
        if prediction < 0:
            prediction = 0
        print(f"The prediction price is {prediction:.2f} for {x} mileage.")
        display_estimate_price(theta_0, theta_1, prediction, x, df)
        accuracy(theta_0, theta_1, prediction, df)
    except AssertionError as e:
        print(f"{AssertionError.__name__}: {e}")
    except ValueError:
        print(f"{ValueError.__name__}: Invalid input.")
    except KeyboardInterrupt:
        print("The program was interrupted. Please try again")
    except OverflowError:
        print("You might change your car.")
    except FileNotFoundError:
        prediction = theta_0 + theta_1 * x
        print(f"{FileNotFoundError.__name__}: "
              f"The prediction price is {prediction:.2f} for {x} mileage.")
    except Exception as e:
        print(f"{type(e).__name__} : Try again.")


if __name__ == "__main__":
    main()
