import pytest
import cv2 

@pytest.fixture
def img_data():
    img_loc='data/sample/1.jpg'
    img = cv2.imread(img_loc)
    return img
