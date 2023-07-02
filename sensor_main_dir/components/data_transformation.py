import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sensor_main_dir.entity.artifect_entity_dataclass import DataTransformationArtifact
from sensor_main_dir.exception import SensorException

from sensor_main_dir.constant_var.training_pipeline import TARGET_COLUMN
from sensor_main_dir.ml.estimator import TargetValueMapping
from sensor_main_dir.utils.main_util import save_object, save_numpy_array_data
from sensor_main_dir.logger_dir import logging

from imblearn.combine import SMOTETomek


class DataTransformation:
    def __init__(self, data_validation_artifact, data_transformation_config):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise SensorException(sys, e)

    @staticmethod
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(sys, e)from e

    @staticmethod
    def get_transform_data_object():
        try:
            logging.info("Entered get_data_transformer_object method of DataTransformation class")
            logging.info("Got numerical cols from schema config")
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            logging.info("Initialized RobustScaler, Simple Imputer")
            preprocessor = Pipeline(steps=[("Imputer", simple_imputer), ("RobustScaler", robust_scaler)])
            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Entered get_data_transformer_object method of DataTransformation class")
            return preprocessor
        except Exception as e:
            raise SensorException(sys, e)from e

    def initiate_data_transformation(self):
        try:

            logging.info("entered in initiate_data_transformation")
            preprocessor = self.get_transform_data_object()
            logging.info("Got processor object")
            train_df = DataTransformation.read_data(file_path=self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(file_path=self.data_validation_artifact.valid_test_file_path)

            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = target_feature_train_df.replace(TargetValueMapping().to_dict())
            logging.info("Got train features and test features of Training dataset")

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())
            logging.info("Got train features and test features of Testing dataset")
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Used the preprocessor object to fit transform the train features")

            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Used the preprocessor object to transform the test features")

            logging.info("Applying SMOTETomek on Training dataset")
            smt = SMOTETomek(sampling_strategy="minority")
            input_feature_train_final_sample, target_feature_train_final_sample = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df)

            logging.info("Applied SMOTETomek to Training dataset")
            logging.info("Applying SMOTETomek on Training dataset")
            input_feature_test_final_sample, target_feature_test_final_sample = smt.fit_resample(input_feature_test_arr,target_feature_test_df)
            logging.info("Applied SMOTETomek to test dataset")

            train_arr = np.c_[input_feature_train_final_sample, target_feature_train_final_sample]
            test_arr = np.c_[input_feature_test_final_sample, target_feature_test_final_sample]

            transformed_object_path = self.data_transformation_config.transformed_object_file_path
            save_object(transformed_object_path, preprocessor)

            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path
            save_numpy_array_data(transformed_train_file_path, array=train_arr)

            transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            save_numpy_array_data(transformed_test_file_path, array=test_arr)
            logging.info("Saved the preprocessor object")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys) from e
