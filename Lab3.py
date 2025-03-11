#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("--file", required=True, help="Path to dataset file")
#args = parser.parse_args()

import pandas as pd

jp_stock_frame = pd.read_csv("JPStockPredict.csv")

print(jp_stock_frame)

def splitData(self):
    pass

def splitDataRandom(self):
    pass