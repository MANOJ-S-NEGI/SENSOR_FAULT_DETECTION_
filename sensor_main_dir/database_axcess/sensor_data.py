import pandas as pd
import numpy as np
import logging
from sensor_main_dir.configuration.mongo_db_connection import MongoDBClient
from sensor_main_dir.constant_var.database import DATABASE_NAME, COLLECTION_NAME
from sensor_main_dir.exception import SensorException
import sys


class SensorData:
    def __init__(self):
        try:
            self.connection_client = MongoDBClient().connection
            logging.info("connection_established to database")
        except Exception as e:
            raise SensorException(e, sys)

    def call_collection_data(self):
        try:
            if DATABASE_NAME is not None:
                DB_collection = self.connection_client[DATABASE_NAME][COLLECTION_NAME]
                logging.info(f'1DATABASE_NAME:{DATABASE_NAME} COLLECTION_NAME1:{COLLECTION_NAME}')
            else:
                raise Exception("Database Name missing")
            return DB_collection
        except Exception as e:
            raise SensorException(e, sys)


    def fetching_collection_data(self):
        try:
            collection = self.call_collection_data()
            records = []
            for i in collection.find():
                records.append(i)
            df = pd.DataFrame(records)
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"sample data from are {df.head(2)}")

            return df

        except Exception as e:
            raise SensorException(e, sys)

SensorData().fetching_collection_data()