# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Uses native python set as a cache for
faster access and unique elements.

Representation:
Set( ((dict_key, value), (dict_key, value)),
      ((dict_key, value), (dict_key, value))
    )
Usage: python -m face_recog.simple_cache
"""
# ===================================================

import sys
from typing import Dict, List, Tuple

from face_recog.exceptions import InvalidCacheInitializationData, NotADictionary
from face_recog.in_memory_cache import InMemoryCache
from face_recog.logger import LoggerFactory

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


class SimpleCache(InMemoryCache):
    """Uses native python set as a cache for
    faster access and unique elements.

    Representation:
    Set( ((dict_key, value), (dict_key, value)),
        ((dict_key, value), (dict_key, value))
        )
    """

    def __init__(self, data: List[Dict] = None):
        """Constructor

        Args:
            data (List[Dict], optional): Initial data to load in the cache.
                Defaults to None.

        Raises:
            InvalidCacheInitializationData: [description]
        """
        if data is not None and (
            type(data) is not list or (len(data) and type(data[0]) is not dict)
        ):
            raise InvalidCacheInitializationData

        self.facial_data = set()
        if data:
            for face_data in data:
                # Add serialized data: ( (dict_key, value), (dict_key, value) )
                self.facial_data.add(self.serialize_dict(face_data))

    def add_data(self, face_data: Dict):
        """Adds facial data to cache.

        Args:
            face_data (Dict): [description]
        """
        # Dict is mutable and hence unhashable. Set
        # doesn't support unhashable data. So we convert the
        # data to immutable tuple. Also with .items() the order
        # might change for the same items so we use the sorted items
        facial_data = self.serialize_dict(face_data)
        self.facial_data.add(facial_data)

    def get_all_data(self) -> List[Dict]:
        """Returns a list of facial data of all the
        registered users from cache.

        Returns:
            List[Dict]: [description]
        """
        return self.deserialize_data(self.facial_data)

    def delete_data(self, face_id: str):
        """Deletes the facial record that match the facial
        Id from cache.

        Args:
            face_id (str): Identifier used for searching

        Returns:
            Deletion status: [description]
        """
        for data in self.facial_data:
            for key_pair in data:
                if face_id in key_pair:
                    self.facial_data.discard(data)
                    return True
        return False

    def serialize_dict(self, data: Dict) -> Tuple[Tuple]:
        """Serializes the dict data so that it can be stored
        in python Set data structure. All mutable data types
        are converted to immutable type.

        Args:
            data (Dict): Facial data

        Raises:
            NotADictionary: [description]

        Returns:
            Tuple[Tuple]: Data safe compatible with Python Set
        """
        # Convert list to tuple
        if "encoding" in data and type(data["encoding"]) is list:
            data["encoding"] = tuple(data["encoding"])
        if type(data) is not dict:
            raise NotADictionary
        return tuple(sorted(data.items()))

    def deserialize_data(self, data) -> List[Dict]:
        """Used for deserializing data.
        Deserialzed form: [ {}, {} ]

        Args:
            data ([type]): [description]

        Returns:
            List[Dict]: [description]
        """
        facial_data = []
        for face_data in data:
            facial_data.append({item[0]: item[1] for item in face_data})

        return facial_data

    def get_details(self) -> List[Dict]:
        """Returns a list of name and unique face id of each registered user

        Returns:
            List[Dict]: [description]
        """
        facial_data = self.get_all_data()
        facial_data = [
            (face_data["id"], face_data["name"]) for face_data in facial_data
        ]
        return facial_data


if __name__ == "__main__":

    # # Save data
    # face_data = {"name": "test3", "encoding": (-3.4, 0.3, -0.823, 1)}
    # ob = SimpleCache(
    #     [
    #         {"name": "test1", "encoding": (-3.4, 0.3, -0.823, 1)},
    #         {"name": "test2", "encoding": (-3.4, 0.3, -0.823, 1)},
    #     ]
    # )
    # print(ob.get_all_data())
    # # [{'encoding': (-3.4, 0.3, -0.823, 1), 'name': 'test2'}, {'encoding': (-3.4, 0.3, -0.823, 1), 'name': 'test1'}]

    # ob.add_data(face_data=face_data)
    # print(ob.get_all_data())
    # # print(sorted(ob.get_all_data(), key=lambda x: x['name']))

    # # output
    # # [{'encoding': (-3.4, 0.3, -0.823, 1), 'name': 'test2'}, {'encoding': (-3.4, 0.3, -0.823, 1), 'name': 'test1'}]
    # serialized_data = (("encoding", (-3.4, 0.3, -0.823, 1)), ("name", "test2"))

    # ob.delete_data(face_id="test1")
    # print(ob.get_all_data())
    pass
