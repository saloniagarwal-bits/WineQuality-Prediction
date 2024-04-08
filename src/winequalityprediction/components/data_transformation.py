import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.winequalityprediction.exception import CustomException
from src.winequalityprediction.logger import logging
from src.winequalityprediction.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform_object(self):
        '''
        this function will transform the data
        '''
        try:
            num_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
            num_pipeline = Pipeline(
                steps= [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info(f'Numerical columns {num_cols}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_cols)
                ]
            )

            return preprocessor
            
        except Exception as ex:
            raise CustomException(ex, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test file done')
            preprocessor_obj = self.get_data_transform_object()

            input_train_df = train_df.drop(['quality'], axis=1)
            target_train_df = train_df['quality']

            input_test_df = test_df.drop(['quality'], axis=1)
            target_test_df = test_df['quality']

            logging.info('Apply preprocessing on train and test df')

            input_train_arr = preprocessor_obj.fit_transform(input_train_df)
            input_test_arr = preprocessor_obj.transform(input_test_df)

            # concatenate the preprocessed input and output features
            train_arr = np.c_[
                input_train_arr, np.array(target_train_df)
            ]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(file_path= self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessor_obj)
            
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as ex:
            raise CustomException(ex, sys)
