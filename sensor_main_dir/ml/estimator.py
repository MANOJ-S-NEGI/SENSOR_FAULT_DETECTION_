import sys, os
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from sensor_main_dir.logger_dir import logging
from sensor_main_dir.exception import SensorException
from sensor_main_dir.constant_var.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        create_dictionary = zip(mapping_response.values(), mapping_response.keys())
        return dict(create_dictionary)


class SensorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object #pickel_file
        self.trained_model_object = trained_model_object

    def predict(self, dataframe):
        logging.info("Entered predict method of SensorTruckModel class")
        try:
            logging.info("Using the trained model to get predictions")
            transformed_feature = self.preprocessing_object.transform(dataframe)
            logging.info("Used the trained model to get predictions")
            prediction = self.trained_model_object.predict(transformed_feature)
            return prediction
        except Exception as e:
            raise SensorException(e, sys) from e




