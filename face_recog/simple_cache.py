# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Uses native python set as a cache for
faster access and unique elements.

Representation:
Set( ((dict_key, value), (dict_key, value)),
      ((dict_key, value), (dict_key, value))
    )
Usage: python -m face_recog.simple_cache
'''
# ===================================================

from face_recog.in_memory_cache import InMemoryCache
from face_recog.exceptions import (InvalidCacheInitializationData,
                                NotADictionary)
from typing import List, Dict, Tuple  

class SimpleCache(InMemoryCache):
    def __init__(self, data:List[Dict]=None):
        if data is not None and (type(data) is not list \
            or (len(data) and type(data[0]) is not dict)):
            raise InvalidCacheInitializationData

        self.facial_data = set()
        if data:
            for face_data in data:
                # Add serialized data: ( (dict_key, value), (dict_key, value) )
                self.facial_data.add(self.serialize_dict(face_data))

    def add_data(self, face_data:Dict):
        # Dict is mutable and hence unhashable. Set
        # doesn't support unhashable data. So we convert the
        # data to immutable tuple. Also with .items() the order 
        # might change for the same items so we use the sorted items
        facial_data = self.serialize_dict(face_data)
        self.facial_data.add(facial_data)


    def get_all_data(self) -> List[Dict]:
        return self.deserialize_data(self.facial_data)


    def delete_data(self, face_id:str):
        for data in self.facial_data:
            for key_pair in data:            
                if face_id in key_pair:
                    self.facial_data.discard(data)
                    return True
        return False


    def serialize_dict(self, data:Dict) -> Tuple[Tuple]:
        # Convert list to tuple
        if 'encoding' in data and \
            type(data['encoding']) is list:
            data['encoding'] = tuple(data['encoding'])
        if type(data) is not dict:
            raise NotADictionary
        return tuple(sorted(data.items()))


    def deserialize_data(self, data) -> List[Dict]:
        """ Deserialzed form: [ {}, {} ]"""
        facial_data = []
        for face_data in data:
            facial_data.append({item[0]: item[1] for item in face_data})

        return facial_data


    def get_details(self) -> List[Dict]:
        """ Returns the name and unique face id of each registered user"""
        facial_data = self.get_all_data()
        facial_data = [(face_data['id'], face_data['name']) \
                        for face_data in facial_data]
        return facial_data


if __name__ == "__main__":
    import numpy as np

    # Save data
    face_data = {'name': 'test3', 
            'encoding': (-3.4, 0.3, -.823, 1)}
    ob = SimpleCache([{'name': 'test1', 'encoding': (-3.4, 0.3, -0.823, 1)},
                     {'name': 'test2', 'encoding': (-3.4, 0.3, -0.823, 1)}])
    print(ob.get_all_data())
    # [{'encoding': (-3.4, 0.3, -0.823, 1), 'name': 'test2'}, {'encoding': (-3.4, 0.3, -0.823, 1), 'name': 'test1'}]
    
    ob.add_data(face_data=face_data)
    print(ob.get_all_data())
    # print(sorted(ob.get_all_data(), key=lambda x: x['name']))
    
    # output
    # [{'encoding': (-3.4, 0.3, -0.823, 1), 'name': 'test2'}, {'encoding': (-3.4, 0.3, -0.823, 1), 'name': 'test1'}]
    serialized_data = (('encoding', (-3.4, 0.3, -0.823, 1)), ('name', 'test2'))

    ob.delete_data(face_id='test1')
    print(ob.get_all_data())
