import os
import sys
from src.winequalityprediction.logger import logging
from src.winequalityprediction.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
#here we can define some of the paramters

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','winequality-red.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Read csv started')
            df = pd.read_csv(self.ingestion_config.raw_data_path)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Read csv done')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as ex:
            raise CustomException(ex,sys)

