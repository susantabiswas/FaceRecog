from face_recog.exceptions import DatabaseFileNotFound
import pytest
from face_recog.json_persistent_storage import JSONStorage
import os
import numpy as np


class TestJSONPersistentStorage:
    @classmethod
    def setup_class(cls):
        # Set the db file path for test usage
        cls.test_db_loc = "data/test_facial_data.json"

    # Temp test file deleted after each test function
    @classmethod
    def teardown_method(cls):
        # remove the test file
        if os.path.exists(cls.test_db_loc):
            os.remove(TestJSONPersistentStorage.test_db_loc)
            print("[TEST] Teardown started: DB file removed!!")

    def test_add_data(self, face_data1, face_data2):
        """ Check if data gets added or not"""
        ob = JSONStorage(db_loc=TestJSONPersistentStorage.test_db_loc)
        ob.add_data(face_data1)
        ob.add_data(face_data2)
        print(ob.get_all_data())
        assert sorted(ob.get_all_data(), key=lambda x: x["name"]) == sorted(
            [face_data1, face_data2], key=lambda x: x["name"]
        )

    def test_missing_db_get_all_data(self, face_data1):
        """Check if an exception is thrown when
        db file is missing during a fetch"""
        ob = JSONStorage(db_loc=TestJSONPersistentStorage.test_db_loc)
        # Add some data and save to disk
        ob.add_data(face_data1)

        # remove the db file
        os.remove(TestJSONPersistentStorage.test_db_loc)
        print("[TEST] DB file removed!!")

        with pytest.raises(DatabaseFileNotFound):
            ob.get_all_data()

    def test_delete_data(self, face_data1, face_data2):
        """ Check if data deletion works"""
        ob = JSONStorage(db_loc=TestJSONPersistentStorage.test_db_loc)
        ob.add_data(face_data1)
        ob.add_data(face_data2)
        ob.delete_data(face_id="test1")

        assert ob.get_all_data() == [face_data2]
