import pymongo
import os
import sys
import certifi

from sensor_main_dir.constant_var.env_variable import MONGODB_URL_KEY
from sensor_main_dir.exception import SensorException
from sensor_main_dir.constant_var.database import DATABASE_NAME

ca = certifi.where()


class MongoDBClient:
    def __init__(self):
        try:
            mongo_db_url = os.getenv(MONGODB_URL_KEY)
            if mongo_db_url is None:
                raise Exception(f"Environment key: is not set.")
            self.connection = pymongo.MongoClient(mongo_db_url)
            #self.connection = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            #use when connect with cloud server coz  localhost closing the connection

        except Exception as e:
            raise SensorException(e, sys)


"""print(MongoDBClient().connection)"""
