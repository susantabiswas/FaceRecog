# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class to handle saving and retrieving facial data.
The data is saved on disk for peersistence, also an in-memory cache is 
used for quicker look ups.

For persistent storage a JSON file is used and for in memory cache,
native python dictionary is used.
'''
from face_recog.validators import path_exists
from face_recog.exceptions import DatabaseFileNotFound
import os

# ===================================================
class FaceDataStore:
    def __init__(self, persistent_data_loc='data/facial_data.json') -> None:
        # check if the DB file exists
        self.persistent_data_loc = persistent_data_loc

    def load_from_disk(self):
        # check if the DB file exists
        if not path_exists(self.persistent_data_loc):
            raise DatabaseFileNotFound
            

    def save_to_disk(self):
        pass

    def add_to_cache(self):
        pass

    def remove_from_cache(self):
        pass

    def get_cache_data(self):
        pass

    def search_data(self):
        pass

    def search_cache_data(self):
        pass

    def search_disk_data(self):
        pass