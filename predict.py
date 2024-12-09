import pandas as pd


def model_result():
    """Read the csv file of the training model"""
    df = pd.read_csv("model.csv")
    theta_0 = df.iloc[0]["theta_0"]
    theta_1 = df.iloc[0]["theta_1"]
    return theta_0, theta_1


def main():
    try:
        print("Enter the mileage for prediciton: ")
        theta_0 = 0
        theta_1 = 0
        x = int(input())
        if x < 0:
            raise AssertionError("Milage cant't be under zero.")
        theta_0, theta_1 = model_result()
        prediction = theta_0 + theta_1 * x
        if prediction < 0:
            prediction = 0
        print(f"The prediction price is {prediction:.2f} for {x} mileage.")
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
