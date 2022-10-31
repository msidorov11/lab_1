import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import balanced_accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sys
import traceback

from logger import Logger

SHOW_LOG = True

class MultiModel():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.dec_tree_path = os.path.join(self.project_path, "dec_tree.sav")
        self.log_reg_path = os.path.join(self.project_path, "log_reg.sav")
        self.rand_forest_path = os.path.join(self.project_path, "rand_forest.sav")
        self.knn_path = os.path.join(self.project_path, "knn.sav")
        self.svc_path = os.path.join(self.project_path, "svc.sav")
        self.log.info("MultiModel is ready")

    def dec_tree(self, use_config: bool, min_samples_split=40,  max_depth=10, criterion = 'gini', predict=False) -> int:
        if use_config:
            try:
                classifier = DecisionTreeClassifier(min_samples_split = self.config.getint("DEC_TREE", "min_samples_split"), 
                max_depth = self.config.getint("DEC_TREE", "max_depth"),
                criterion = self.config["DEC_TREE"]["criterion"])
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = DecisionTreeClassifier(min_samples_split = min_samples_split, max_depth = max_depth, criterion = criterion)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(balanced_accuracy_score(self.y_test, y_pred))
        params = {
            'min_samples_split': min_samples_split,
            'max_depth': max_depth,
            'criterion': criterion,
            'path': self.log_reg_path
            }
        return self.save_model(classifier, self.dec_tree_path, "DEC_TREE", params)

    def log_reg(self, use_config: bool, C = 2, penalty = 'l2', predict=False) -> int:
        if use_config:
            try:
                classifier = LogisticRegression(C = self.config.getint("LOG_REG", "C"), penalty = self.config["LOG_REG"]["penalty"])
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = LogisticRegression(C = C, penalty = penalty)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(balanced_accuracy_score(self.y_test, y_pred))
        params = {
            'C': C,
            'penalty': penalty,
            'path': self.log_reg_path
            }
        return self.save_model(classifier, self.log_reg_path, "LOG_REG", params)

    def rand_forest(self, use_config: bool, criterion = 'gini', max_depth = 20, n_estimators=60, predict=False) -> int:
        if use_config:
            try:
                classifier = RandomForestClassifier(
                    criterion=self.config["RAND_FOREST"]["criterion"],
                    max_depth=self.config.getint("RAND_FOREST", "max_depth"),
                    n_estimators=self.config.getint("RAND_FOREST", "n_estimators"))
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = RandomForestClassifier(
                criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(balanced_accuracy_score(self.y_test, y_pred))
        params = {'criterion': criterion,
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'path': self.rand_forest_path}
        return self.save_model(classifier, self.rand_forest_path, "RAND_FOREST", params)

    def knn(self, use_config: bool, n_neighbors=3, weights='uniform', p=1, predict=False) -> int:
        if use_config:
            try:
                classifier = KNeighborsClassifier(n_neighbors=self.config.getint(
                    "KNN", "n_neighbors"), weights=self.config["KNN"]["weights"], p=self.config.getint("KNN", "p"))
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights, p=p)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(balanced_accuracy_score(self.y_test, y_pred))
        params = {'n_neighbors': n_neighbors,
                  'weights': weights,
                  'p': p,
                  'path': self.knn_path}
        return self.save_model(classifier, self.knn_path, "KNN", params)

    def svc(self, use_config: bool, C=5, gamma = 0.1, kernel='rbf', predict=False) -> int:
        if use_config:
            try:
                classifier = SVC(C = self.config.getint("SVC", "C"),
                                gamma = self.config.getfloat("SVC", "gamma"),
                                kernel=self.config["SVC"]["kernel"])
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = SVC(C=C, gamma=gamma, kernel=kernel)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(balanced_accuracy_score(self.y_test, y_pred))
        params = {'C': C,
                'gamma': gamma,
                'kernel': kernel,
                'path': self.svc_path}
        return self.save_model(classifier, self.svc_path, "SVC", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    multi_model = MultiModel()
    multi_model.dec_tree(use_config=True, predict=True)
    multi_model.log_reg(use_config=True, predict=True)
    multi_model.rand_forest(use_config=True, predict=True)
    multi_model.knn(use_config=True, predict=True)
    multi_model.svc(use_config=True, predict=True)

