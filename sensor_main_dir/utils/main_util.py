import os
import sys
import dill as dill
import numpy as np
from sensor_main_dir.exception import SensorException
import logging
import yaml


def read_yaml_file(file_path):
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise SensorException(sys, e)


def write_yaml_file(file_path, content):
    try:
        with open(file_path, "w") as file:
            yaml.dump(content, file)  # dump from content to write in file

    except Exception as e:
        raise SensorException(e, sys)


def save_numpy_array_data(file_path, array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise SensorException(e, sys)


def load_numpy_array_data(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SensorException(sys, e)


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise SensorException(sys, e)


def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_object:
            obj = dill.load(file_object)
            logging.info("Exited the load_object method of MainUtils class")
            return obj
    except Exception as e:
        raise SensorException(sys, e)
