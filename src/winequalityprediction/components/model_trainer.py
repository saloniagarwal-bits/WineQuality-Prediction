from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import os, sys
from dataclasses import dataclass
from src.winequalityprediction.utils import save_object, evaluate_models

from src.winequalityprediction.exception import CustomException
from src.winequalityprediction.logger import logging

@dataclass
class ModelTrainerConfig:
    train_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info(' Split train and test data')

            X_train = train_arr[:,:-1]
            X_test = test_arr[:,:-1]
            y_train = train_arr[:,-1]
            y_test = test_arr[:,-1]

            models = {
                'DecisionTree Classifier' : DecisionTreeClassifier(),
                'KNeighbor Classifier' : KNeighborsClassifier(),
                'Random Forest Classifier': RandomForestClassifier(),
                'Adaboost Classifier' : AdaBoostClassifier(),
                'Support Vector Classifier' : SVC(),
                'CatBoost Classifier' : CatBoostClassifier(allow_writing_files=False, silent=True)
                # 'XGB Classifier' : XGBClassifier()
                
            }

            model_report = evaluate_models(X_train , y_train, X_test, y_test , models)
            print(model_report)

            #Get best model score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_obj = models[best_model_name]
            print(best_model_name)

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info('Best model found on train and test dataset')

            save_object(
                file_path= self.model_train_config.train_model_path,
                obj = best_model_obj
            )

            predicted = best_model_obj.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            return acc_score

        except Exception as ex:
            raise CustomException(ex, sys)