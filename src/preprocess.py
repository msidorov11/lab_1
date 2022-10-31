import configparser
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import traceback

from logger import Logger

TEST_SIZE = 0.5
SHOW_LOG = True


class DataMaker():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")
        self.data_path = os.path.join(self.project_path, "seeds.csv")
        self.X_path = os.path.join(self.project_path, "X.csv")
        self.y_path = os.path.join(self.project_path, "y.csv")
        self.train_path = [os.path.join(self.project_path, "X_train.csv"), os.path.join(
            self.project_path, "y_train.csv")]
        self.test_path = [os.path.join(self.project_path, "X_test.csv"), os.path.join(
            self.project_path, "y_test.csv")]
        self.log.info("DataMaker is ready")

    def get_data(self) -> set:
        dataset = pd.read_csv(self.data_path)
        X = pd.DataFrame(dataset[['Area', 'Perimeter', 'Compactness', 'Kernel.Length', 'Kernel.Width', 'Asymmetry.Coeff', 'Kernel.Groove']].values)
        y = pd.DataFrame(dataset['Type'].values)
        X.to_csv(self.X_path, index=True)
        y.to_csv(self.y_path, index=True)
        if os.path.isfile(self.X_path) and os.path.isfile(self.y_path):
            self.log.info("X and y data is ready")
            self.config["DATA"] = {'X_data': self.X_path,
                                   'y_data': self.y_path}
            return (X, y)
        else:
            self.log.error("X and y data is not ready")
            return ()

    def split_data(self, test_size=TEST_SIZE) -> set:
        try:
            X, y = self.get_data()
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0, stratify=y)
        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])
        self.config["SPLIT_DATA"] = {'X_train': self.train_path[0],
                               'y_train': self.train_path[1],
                               'X_test': self.test_path[0],
                               'y_test': self.test_path[1]}
        self.log.info("Train and test data is ready")
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        return (X_train, X_test, y_train, y_test)

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        df = df.reset_index(drop=True)
        df.to_csv(path, index=True)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()