# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class to handle persistent data storage.
This uses simple JSON file to save data.

Usage: python -m face_recog.json_persistent_storage
'''
# ===================================================
import json
from face_recog.persistent_storage import PersistentStorage
from face_recog.exceptions import (DatabaseFileNotFound)
from face_recog.validators import path_exists
import os 
import re 
from typing import Dict 

class JSONStorage(PersistentStorage):
    def __init__(self, db_loc:str='./data/facial_data_db.json'):
        self.db_loc = db_loc
        
    def add_data(self, face_data:Dict):
        data = []
        # check if the db exists, otherwise create one
        base_path, filename = os.path.split(self.db_loc)
            
        if not path_exists(base_path):
            print("[INFO] DB path doesn't exist! Attempting path creation...")
            os.makedirs(base_path)
        if os.path.exists(self.db_loc):
            # load the existing data
            data = self.get_all_data()

        try:
            # Add the new data and save to disk
            data.append(face_data)

            with open(self.db_loc, 'w') as f:
                json.dump(data, f)
            print('[INFO] Data saved to DB...')
        except Exception as exc:
            raise exc


    def get_all_data(self):
        # Data load will fail incase the file doesn't exist
        if not path_exists(self.db_loc):
            raise DatabaseFileNotFound
        try:
            # load the existing data
            with open(self.db_loc, 'r') as f:
                data = json.load(f)
                # deserialize the dict data and convert back to dict
                return data
        except Exception as exc:
            raise exc



if __name__ == "__main__":
    import numpy as np
    ob = JSONStorage(db_loc='data/test_facial_data.json')
    # Save data
    face_data = {'name': 'test2', 
            'encoding': (-3.4, 0.3, -.823, 1)}
    ob.add_data(face_data=face_data)
    print(ob.get_all_data())

    ob.add_data(face_data=face_data)
    print(ob.get_all_data())

    # remove the test file
    os.remove('data/test_facial_data.json')

    
    