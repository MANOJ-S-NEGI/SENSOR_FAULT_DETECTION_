from sensor_main_dir.logger_dir import logging
from sensor_main_dir.entity.artifect_entity_dataclass import ModelPusherArtifact
import os,sys
import shutil
from sensor_main_dir.exception import SensorException


class ModelPusher:
    def __init__(self, model_pusher_config, model_eval_artifact):

        self.model_pusher_config = model_pusher_config
        self.model_eval_artifact = model_eval_artifact

    def initiate_model_pusher(self):
        try:
            trained_model_path = self.model_eval_artifact.trained_model_path
            logging.info(f"Creating model pusher dir to save model")
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(self.model_pusher_config.model_evaluation_dir, exist_ok=True)
            logging.info(f"model pusher dir created")
            shutil.copy(src=trained_model_path, dst=model_file_path)

            logging.info(f"Creating saved Model dir to save model")
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            logging.info(f"saved model dir created")
            shutil.copy(src=trained_model_path, dst=saved_model_path)
            logging.info(f"model saved in directory")
            return model_file_path

        except Exception as e:
            raise SensorException(e, sys)

