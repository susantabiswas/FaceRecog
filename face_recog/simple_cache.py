# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Uses native python set as a cache for
 fast look ups
 
Usage: python -m face_recog.simple_cache
'''
# ===================================================

from face_recog.in_memory_cache import InMemoryCache
from face_recog.exceptions import InvalidCacheInitializationData

class SimpleCache(InMemoryCache):
    def __init__(self, data=None):
        if data and type(data) is not list:
            raise InvalidCacheInitializationData
        # Initialize the cache with data if supplied
        self.facial_data = set(data) or set()

    def add_data(self, face_data):
        self.facial_data.add(face_data)

    def get_all_data(self):
        return list(self.facial_data)


if __name__ == "__main__":
    ob = SimpleCache([1, 2, 4])
    ob.add_data(3)
    print(ob.get_all_data())

    ob.add_data(4)
    print(ob.get_all_data())

    