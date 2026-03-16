import pandas as pd


class AggregatedDataReader:
    def __init__(self):
        self.data = pd.DataFrame()

    def read_data(self, path=None):
        if path is None:
            path = "./AggregatedData/RandomData.csv"
        self.data = pd.read_csv(path)
