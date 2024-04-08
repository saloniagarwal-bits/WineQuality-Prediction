from src.winequalityprediction.logger import logging
from src.winequalityprediction.exception import CustomException
from src.winequalityprediction.components.data_ingestion import DataIngestion
from src.winequalityprediction.components.data_transformation import DataTransformation
from src.winequalityprediction.components.model_trainer import ModelTrainer
import sys

if __name__ == "__main__":
    logging.info('Exceution started')

    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

        model_trainer = ModelTrainer()
        print(f'Best Score is {model_trainer.initiate_model_trainer(train_arr, test_arr)}')

    except Exception as ex:
        logging.info('Custom Exception')
        raise CustomException(ex,sys)