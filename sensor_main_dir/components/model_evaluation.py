import pandas as pd
from sensor_main_dir.entity.artifect_entity_dataclass import ModelEvaluationArtifact

from sensor_main_dir.ml.metric import calculate_metric

from sensor_main_dir.utils.main_util import load_object, write_yaml_file

from sensor_main_dir.logger_dir import logging
from sensor_main_dir.exception import SensorException
from sensor_main_dir.constant_var.training_pipeline import TARGET_COLUMN, MODEL_TRAINER_EXPECTED_SCORE
from sensor_main_dir.ml.estimator import TargetValueMapping
#from sensor_main_dir.ml.estimator import ModelResolver
import sys, os


class ModelEvaluation:
    def __init__(self, model_evaluation_config, data_validation_artifact, model_trainer_artifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)from e

    def initiate_model_evaluation(self):
        try:
            logging.info(f"initiating model evaluation")
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path
            logging.info(f"training file and test file called")
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)
            df = pd.concat([train_df, test_df])
            logging.info(f"Distributing the features and target from dataframe : df")
            y_true = df[TARGET_COLUMN]
            y_true.replace(TargetValueMapping().to_dict(), inplace=True)
            df.drop(TARGET_COLUMN, axis=1, inplace=True)

            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            latest_metric_trainer = load_object(trained_model_file_path)
            evaluated_data = calculate_metric(model=latest_metric_trainer, x=df, y=y_true)
            if evaluated_data.f1_score < MODEL_TRAINER_EXPECTED_SCORE:
                raise Exception(f"Evaluated score is lower than Expected Score")

            model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=None, improved_accuracy=None, best_model_path=None, trained_model_path=trained_model_file_path, train_model_metric_artifact=None, best_model_metric_artifact=latest_metric_trainer)

            model_eval_report = model_evaluation_artifact.__dict__
            logging.info(f"creating the path to save the report")
            os.makedirs(self.model_evaluation_config.model_evaluation_dir)
            write_yaml_file(self.model_evaluation_config.report_file_path, model_eval_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            logging.info(f"latest model accuracy obtained is {evaluated_data.f1_score}")
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)



