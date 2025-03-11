#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("--file", required=True, help="Path to dataset file")
#args = parser.parse_args()

import csv

with open('JPStockPredict.csv', 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    print("Header:", header)
    for row in csvreader:
        print("Row:", row)
        # Process data here

def splitData(self):

def splitDataRandom(self):