#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("--file", required=True, help="Path to dataset file")
#args = parser.parse_args()

import pandas as pd
import math

jp_stock_frame = pd.read_csv("JPStockPredict.csv")

# print(jp_stock_frame)

def splitData(data_frame):
    #  80% training, 20% testing.
    rows = len(data_frame)
    training = data_frame.head(math.floor(rows * 0.8))
    testing = data_frame.tail(math.floor(rows * 0.2))
    return (training, testing)

def splitDataRandom(self):
    pass


if __name__ == "__main__":
    training, testing = splitData(jp_stock_frame)
    print(training)
    print(testing)