# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class to handle saving and retrieving facial data.
The data is saved on disk for peersistence, also an in-memory cache is 
used for quicker look ups.

For persistent storage a JSON file is used and for in memory cache,
native python set is used.
Cache doesn't allow duplicates. Without duplicates, entries for 
search are limited.

Cache is initialized with data loaded from DB. Then each addition 
of face data for a new user goes through two writes:
1. Addition to cache
2. Saved to disk DB.

Cache is always up to date with the latest facial data.

Usage: python -m face_recog.face_data_store
'''
from face_recog.json_persistent_storage import JSONStorage
from face_recog.simple_cache import SimpleCache
from face_recog.validators import path_exists
from face_recog.exceptions import DatabaseFileNotFound, InvalidCacheInitializationData
import os

# ===================================================
class FaceDataStore:
    def __init__(self, persistent_data_loc='data/facial_data.json') -> None:
        # persistent storage handler
        self.db_handler = JSONStorage(persistent_data_loc)
        saved_data = []
        try:
            # Initialize the cache handler with data from DB
            saved_data = self.db_handler.get_all_data()
            print('[INFO] Data loaded from DB with {} entries...'.format(len(saved_data)))
        except DatabaseFileNotFound:
            print('[INFO] No existing DB file found!!')
        try:
            self.cache_handler = SimpleCache(saved_data)
        except InvalidCacheInitializationData:
            raise InvalidCacheInitializationData

    def add_facial_data(self, facial_data):
        # add to cache and save on disk
        self.cache_handler.add_data(face_data=facial_data)
        self.db_handler.add_data(face_data=facial_data)

    def remove_facial_data(self, face_id=None):
        self.cache_handler.delete_data(face_id=face_id)
        self.db_handler.delete_data(face_id=face_id)
    
    def get_all_facial_data(self):
        # Since all face
        return self.cache_handler.get_all_data()


if __name__ == "__main__":
    import numpy as np

    print('Saving 1 entry to DataStore...')
    ob = FaceDataStore(persistent_data_loc='data/test_facial_data.json')
    # Save data
    face_data = {'name': 'test2', 
                'encoding': (-3.4, 0.3, -.823, 1)}
    ob.add_facial_data(face_data)
    print(ob.get_all_facial_data())

    print('Saving 2 entry to DataStore with existing DB(1 entry)...')
    # Now we again create a DB and again add the same data
    # cache will return only 2 entry but DB will have both
    ob1 = FaceDataStore(persistent_data_loc='data/test_facial_data.json')
    # Save data
    face_data1 = {'name': 'test1', 
            'encoding': (-3.4, 0.3, -.823, 1)}
    face_data2 = {'name': 'test2', 
            'encoding': (-3.4, 0.3, -.823, 1)}
    ob1.add_facial_data(face_data1)
    ob1.add_facial_data(face_data2)

    print('DB data',ob1.db_handler.get_all_data())
    print('Cache data', ob1.cache_handler.get_all_data())
    print('API data', ob1.get_all_facial_data())

    # remove entry
    print('\n\nAfter Deletion ******')
    ob1.remove_facial_data(face_id="test2")
    print('DB data',ob1.db_handler.get_all_data())
    print('Cache data', ob1.cache_handler.get_all_data())
    print('API data', ob1.get_all_facial_data())

    # remove the test file
    os.remove('data/test_facial_data.json')
    print('File: {} deleted!'.format('data/test_facial_data.json'))
    