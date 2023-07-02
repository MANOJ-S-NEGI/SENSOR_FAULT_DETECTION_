from sensor_main_dir.pipeline.training import TrainPipeLine
from sensor_main_dir.logger_dir import logging
from sensor_main_dir.exception import SensorException
import sys
if __name__ == "__main__":
    try:
        train_pipe_line = TrainPipeLine()
        train_pipe_line.run_pipeline()
        print("pipeline_working_fine")
    except Exception as e :
        raise SensorException(e, sys)from e

