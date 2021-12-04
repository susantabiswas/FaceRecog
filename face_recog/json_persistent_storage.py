# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class to handle persistent data storage.
This uses simple JSON file to save data.

Usage: python -m face_recog.json_persistent_storage
"""
# ===================================================
import json
import os
import sys
from typing import Dict, List

from face_recog.exceptions import DatabaseFileNotFound
from face_recog.logger import LoggerFactory
from face_recog.persistent_storage import PersistentStorage
from face_recog.validators import path_exists

# Load the custom logger
logger = None
try:
    logger_ob = LoggerFactory(logger_name=__name__)
    logger = logger_ob.get_logger()
    logger.info("{} loaded...".format(__name__))
    # set exception hook for uncaught exceptions
    sys.excepthook = logger_ob.uncaught_exception_hook
except Exception as exc:
    raise exc


class JSONStorage(PersistentStorage):
    """Class to handle persistent data storage.
    This uses simple JSON file to save data.

    Args:
        PersistentStorage ([type]): [description]
    """

    def __init__(self, db_loc: str = "./data/facial_data_db.json"):
        """Constructor

        Args:
            db_loc (str, optional): Path to save the persistent DB. Defaults to './data/facial_data_db.json'.
        """
        self.db_loc = db_loc

    def add_data(self, face_data: Dict):
        """Adds facial data to persistent DB.

        Args:
            face_data (Dict): [description]
        """
        data = []
        # check if the db exists, otherwise create one
        base_path, filename = os.path.split(self.db_loc)

        if not path_exists(base_path):
            logger.info("DB path doesn't exist! Attempting path creation...")
            os.makedirs(base_path)
        if os.path.exists(self.db_loc):
            # load the existing data
            data = self.get_all_data()
        try:
            # Add the new data and save to disk
            data.append(face_data)
            self.save_data(data=data)
            logger.info("Data saved to DB...")
        except Exception as exc:
            raise exc

    def get_all_data(self) -> List:
        """Returns a list of facial data of all the
        registered users from cache.

        Returns:
            List[Dict]: [description]
        """
        # Data load will fail incase the file doesn't exist
        if not path_exists(self.db_loc):
            raise DatabaseFileNotFound
        try:
            # load the existing data
            with open(self.db_loc, "r") as f:
                data = json.load(f)
                # convert the list to tuple to keep
                # consistency across
                return self.sanitize_data(data)
        except Exception as exc:
            raise exc

    def delete_data(self, face_id: str) -> bool:
        """Deletes the facial record that match the facial
        Id from persistent DB.

        Args:
            face_id (str): Identifier used for searching

        Returns:
            Deletion status: [description]
        """
        # load and search if face id exists and
        # save the data without that entry
        all_data = self.get_all_data()
        num_entries = len(all_data)
        for idx, face_data in enumerate(all_data):
            for key_pair in face_data.items():
                # Check if the face id exists in current data item
                if face_id in key_pair:
                    all_data.pop(idx)

        if num_entries != len(all_data):
            self.save_data(data=all_data)
            logger.info(
                ("{} face(s) deleted and updated" " data saved to DB...").format(
                    num_entries - len(all_data)
                )
            )
            return True
        return False

    def save_data(self, data: Dict = None):
        """Saves data in the persistent DB.

        Args:
            data (Dict, optional): Facial Data. Defaults to None.
        """
        if data is not None:
            with open(self.db_loc, "w") as f:
                json.dump(data, f)

    def sanitize_data(self, data: List[Dict]) -> List[Dict]:
        """Converts data types so that the data types
        are consistent across the system.
        Facial encodings are initially in numpy format,
        which is then changed to a tuple so that it can be saved to
        cache, this method ensures that the data that is loaded
        back from JSON file also has tuple for the encoding instead of a list.

        Args:
            data (Dict): Facial Data

        Returns:
            [type]: [description]
        """
        for face_data in data:
            face_data["encoding"] = tuple(face_data["encoding"])
        return data


if __name__ == "__main__":
    """ Sanity checks """

    # ob = JSONStorage(db_loc="data/test_facial_data.json")
    # # Save data
    # face_data1 = {"name": "test1", "encoding": (-3.4, 0.3, -0.823, 1)}
    # face_data2 = {"name": "test2", "encoding": (-3.4, 0.3, -0.823, 1)}
    # face_data3 = {"name": "test3", "encoding": (-3.4, 0.3, -0.823, 1)}

    # ob.add_data(face_data=face_data1)
    # print(ob.get_all_data())

    # ob.add_data(face_data=face_data2)
    # print(ob.get_all_data())

    # # remove data
    # ob.delete_data(face_id="test1")
    # print("Test1 data deleted")
    # print(ob.get_all_data())

    # # remove the test file
    # os.remove("data/test_facial_data.json")
    # print("File: {} deleted!".format("data/test_facial_data.json"))
    pass