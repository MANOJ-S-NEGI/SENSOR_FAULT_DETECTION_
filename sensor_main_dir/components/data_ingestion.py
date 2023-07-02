import pandas as pd
from sensor_main_dir.logger_dir import logging
import os,sys
from sklearn.model_selection import train_test_split
from sensor_main_dir.constant_var.training_pipeline import SCHEMA_FILE_PATH, SCHEMA_DROP_COLUMNS
from sensor_main_dir.database_axcess.sensor_data import SensorData
from sensor_main_dir.entity.config_entity import DataIngestionConfig
from sensor_main_dir.exception import SensorException
from sensor_main_dir.utils.main_util import read_yaml_file
from sensor_main_dir.entity.artifect_entity_dataclass import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(sys, e)

    def export_data_feature_store(self):
        """
        Method      : export_data_feature_stor
        Description : export the data from server to dataframe
        version     :1.0.0
                    :return:
                            output      : save the database record to the mentioned location
                            on_failure  : write exception log and then raise an exception
        """
        try:
            dataframe = SensorData().fetching_collection_data()
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise SensorException(sys, e)

    def split_data_as_train_test(self, dataframe):
        """
        Method      : split_data_as_train_test
        Description : splits train and test from dataset
        version     :1.0.0
        :return:
                output      : "split the train test in their destination"
                on_failure  : write exception log and then raise an exception
        """
        try:
            logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

            split_size = self.data_ingestion_config.train_test_split_ratio

            training_file_path = self.data_ingestion_config.training_file_path
            training_data_dir = os.path.dirname(training_file_path)

            test_file_path = self.data_ingestion_config.testing_file_path
            test_data_dir = os.path.dirname(test_file_path)

            train_set, test_set = train_test_split(dataframe, random_state=13, test_size=split_size)

            logging.info(f"Training and test data split initiated")
            os.makedirs(training_data_dir, exist_ok=True)
            train_set.to_csv(training_file_path, index=False, header=True)

            os.makedirs(test_data_dir, exist_ok=True)
            test_set.to_csv(test_file_path, index=False, header=True)
            logging.info(f"training and test data split completed")

        except Exception as e:
            raise (sys, e)

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_data_feature_store()
            _schema_config_ = read_yaml_file(SCHEMA_FILE_PATH)
            dataframe = dataframe.drop(_schema_config_[SCHEMA_DROP_COLUMNS], axis=1)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise(e, sys)

