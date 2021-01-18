import pytest
from face_recog.face_data_store import FaceDataStore
import os


class TestFaceDataStore:
    @classmethod
    def setup_class(cls):
        # Set the db file path for test usage
        cls.test_db_loc = "data/test_facial_data.json"

    # Temp test file deleted after each test function
    @classmethod
    def teardown_method(cls):
        # remove the test file
        if os.path.exists(cls.test_db_loc):
            os.remove(TestFaceDataStore.test_db_loc)
            print("[TEST] Teardown started: DB file removed!!")

    def test_add_data(self, face_data1, face_data2):
        """ Check if data gets added or not"""
        ob = FaceDataStore(persistent_data_loc=TestFaceDataStore.test_db_loc)
        ob.add_facial_data(face_data1)
        ob.add_facial_data(face_data2)
        print(ob.get_all_facial_data())
        assert sorted(ob.get_all_facial_data(), key=lambda x: x["name"]) == sorted(
            [face_data1, face_data2], key=lambda x: x["name"]
        )

    def test_delete_data(self, face_data1, face_data2):
        """ Check if data deletion works"""
        ob = FaceDataStore(persistent_data_loc=TestFaceDataStore.test_db_loc)
        ob.add_facial_data(face_data1)
        ob.add_facial_data(face_data2)
        ob.remove_facial_data(face_id="test1")

        assert sorted(ob.get_all_facial_data(), key=lambda x: x["name"]) == [face_data2]
