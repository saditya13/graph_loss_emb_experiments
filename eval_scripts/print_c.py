import pandas as pd
def print_classes():
    val = pd.read_csv('../val.csv')
    print(val.shape())
    print(val.value_counts())


print_classes()