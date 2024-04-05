from src.winequalityprediction.logger import logging
from src.winequalityprediction.exception import CustomException
from src.winequalityprediction.components.data_ingestion import DataIngestion
import sys

if __name__ == "__main__":
    logging.info('Exceution started')

    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

    except Exception as ex:
        logging.info('Custom Exception')
        raise CustomException(ex,sys)