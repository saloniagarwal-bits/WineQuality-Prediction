import os, sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from src.winequalityprediction.exception import CustomException
from src.winequalityprediction.logger import logging
import pandas as pd
import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path , 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as ex:
        raise CustomException(ex, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as ex:
        raise CustomException(ex, sys)


