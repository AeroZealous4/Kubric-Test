import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"

def Wt_Caculation():

    response = requests.get(TRAIN_DATA_URL)  
    # print(response)
    # Train = Train.to_numpy();
    # print(Train);

    Train = numpy.genfromtxt("linreg_train.csv",delimiter=',')
    # print(Train.shape);
    X_Data = Train[0][1:];
    Y_Data = Train[1][1:];
    # print(X_Data.shape)
    # print(Y_Data.shape)
    X_Data = X_Data.reshape(266,1)
    Y_Data = Y_Data.reshape(266,1)

    X_Data = numpy.hstack((numpy.ones((266,1)), X_Data));
    # print(X_Data.shape)
    
    b = numpy.linalg.inv(X_Data.T @ X_Data) @ X_Data.T @ Y_Data;
    print(b)

def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    Wt_Caculation();

    # YOUR IMPLEMENTATION HERE
    ...
    return area*0.0345 + 1153.1


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
