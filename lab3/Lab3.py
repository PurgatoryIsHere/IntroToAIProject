import pandas as pd
import argparse
import math

def splitData(data_frame):
    #  80% training, 20% testing.
    rows = len(data_frame)
    training = data_frame.head(math.floor(rows * 0.8))
    testing = data_frame.tail(math.floor(rows * 0.2))
    return (training, testing)

def splitDataRandom(data_frame):

    data_frame = data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
    
    rows = len(data_frame)
    
    training = data_frame.head(math.floor(rows * 0.8))
    testing = data_frame.tail(rows - math.floor(rows * 0.8))
    
    return training, testing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", required=True, help="Path to dataset file")
    args = parser.parse_args()

    df = pd.read_csv(args.filename)

    

    print("\n===Statistics==")
    print(f"Mean:\n{df[['open', 'close', 'high', 'low', 'volume', 'adj_close']].mean()}\n")
    print(f"Max:\n{df[['open', 'close', 'high', 'low', 'volume', 'adj_close']].max()}\n")
    print(f"Min:\n{df[['open', 'close', 'high', 'low', 'volume', 'adj_close']].min()}\n")
    print(f"Correlation:\n{df[['open', 'close', 'high', 'low', 'volume', 'adj_close']].corr()}\n")


    print("\nSplitting data no random\n")
    training, testing = splitData(df)
    print("\nTraining Data\n")
    print(training)
    print("\nTesting Data\n")
    print(testing)

    print("\nSplitting data randomly\n")
    randTrain, randTest = splitDataRandom(df)
    print("\nTraining Data\n")
    print(randTrain)
    print("\nTesting Data\n")
    print(randTest)