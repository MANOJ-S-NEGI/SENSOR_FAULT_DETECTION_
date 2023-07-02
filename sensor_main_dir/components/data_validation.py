import json
import os
import sys
import pandas as pd
import scipy
from scipy.stats import ks_2samp
import numpy as np
#from evidently.model_profile.sections import DataDriftProfileSection
from sensor_main_dir.constant_var.training_pipeline import SCHEMA_FILE_PATH
from sensor_main_dir.entity.artifect_entity_dataclass import DataValidationArtifact
from sensor_main_dir.exception import SensorException
from sensor_main_dir.logger_dir import logging
from sensor_main_dir.utils.main_util import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact, data_validation_config):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e,sys)from e

    def validate_number_of_columns(self, dataframe):
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise SensorException(e, sys)

    def is_numerical_column_exist(self, dataframe):
        """This function check numerical column is present in dataframe or not"""
        try:
            dataframe_columns = dataframe.columns
            status = True
            missing_numerical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    status = False
                    missing_numerical_columns.append(column)
            logging.info(f"Missing numerical column:{len(missing_numerical_columns)} {missing_numerical_columns}")
            return status

        except Exception as e:
            raise SensorException(e, sys) from e

    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e, sys)

    def detect_dataset_drift(self, reference_df, current_df, threshold=0.5):
        try:
            report_data_drift = {}
            for col in reference_df.columns:
                data1 = reference_df[col]
                data2 = current_df[col]
                drift = scipy.stats.ks_2samp(data1, data2)

                if drift.pvalue >= threshold:
                    continue
                else:
                    report_data_drift.update({col: {"status": "True", "p-value": f"{drift.pvalue:.2f}"}})
            report = report_data_drift
            data_drift_file = self.data_validation_config.drift_report_file_path
            data_drift_path = os.path.dirname(data_drift_file)
            os.makedirs(data_drift_path, exist_ok=True)
            write_yaml_file(file_path=data_drift_file, content=report_data_drift)
            logging.info(f"Report saved in Log_Dir")

            if report is not None:
                status = True
            else:
                status = False

            return status

        except Exception as e:
            raise SensorException(e, sys)from e

    """try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference_df, current_df
            report = data_drift_profile.json()
            json_report = json.loads(report),

            file_path=self.data_validation_config.drift_report_file_path,)
            write_yaml_file(file_path=data_drift_file, content = json_report, replace=True)
            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]
            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
    """

    def initiate_data_validation(self):
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        Output      :   Returns bool value based on validation results
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_numerical_column_exist(dataframe=train_df)
            logging.info(f"All numerical columns present in training dataframe")
            if not status:
                validation_error_msg += f"Numerical columns are missing in training dataframe."

            status = self.is_numerical_column_exist(dataframe=test_df)
            logging.info(f"All numerical columns present in testing dataframe")
            if not status:
                validation_error_msg += f"Numerical columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                logging.info(f"validation_status:{validation_status}")

                if drift_status:
                    logging.info(f"Drift detected.")

            else:
                logging.info(f"Validation_error: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path="None",
                invalid_test_file_path="None",
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys) from e