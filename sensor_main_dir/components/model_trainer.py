
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys, os
import xgboost
from sklearn.model_selection import GridSearchCV


from sensor_main_dir.entity.artifect_entity_dataclass import ModelTrainerArtifact
from sensor_main_dir.ml.estimator import SensorModel
from sensor_main_dir.constant_var.training_pipeline import MODEL_TRAINER_EXPECTED_SCORE, MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
from sensor_main_dir.ml.metric import calculate_metric
from sensor_main_dir.logger_dir import logging
from sensor_main_dir.exception import SensorException
from sensor_main_dir.utils.main_util import load_numpy_array_data, save_object, load_object


class ModelTrainer:
    def __init__(self, data_transformation_artifact, model_trainer_config):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise SensorException (e, sys) from e

    @property
    def grid_parameters(self):
        try:
            para_grid = {"n_estimators": [81], "max_depth": [5]}
            return para_grid
        except Exception as e:
            raise SensorException(e, sys)from e

    def training_model(self):
        try:
            parameters = self.grid_parameters
            XGBOOST_ = xgboost.XGBClassifier()
            grid_cv_with_xgboost = GridSearchCV(estimator=XGBOOST_, param_grid=parameters, cv=3, verbose=2, n_jobs=1)
            return grid_cv_with_xgboost
        except Exception as e:
            raise SensorException(e, sys)from e

    def initiate_model_trainer(self):
        try:
            model = self.training_model()
            logging.info("Initiating the Model_Trainer..")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train, x_test, y_test = (train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1],)
            train_model = model.fit(x_train, y_train)
            if train_model.best_score_ < MODEL_TRAINER_EXPECTED_SCORE:
                raise f"Model Score is below then expected score retain the model"
            train_classification_metric = calculate_metric(train_model, x=x_train, y=y_train)
            test_classification_metric = calculate_metric(train_model, x=x_test, y=y_test)

            if train_classification_metric.f1_score - test_classification_metric.f1_score > MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD:
                raise f"Retrain your model as model is over_fitting or under_fitting"
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            logging.info(f"model obtained accuracy Score is:{train_model.best_score_:.2f}")

            file_path = self.model_trainer_config.trained_model_file_path
            dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(dir_path, exist_ok=True)
            sensor_model = SensorModel(preprocessing_object=preprocessor, trained_model_object=train_model)
            save_object(file_path=file_path, obj=sensor_model)

            Model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=file_path,
                                                          train_metric_artifact=train_classification_metric,
                                                          test_metric_artifact=test_classification_metric)
            return Model_trainer_artifact

        except Exception as e:
            raise (e, sys)
