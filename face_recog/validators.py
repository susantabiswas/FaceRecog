import os


def is_valid_img(image):
    return image is None or \
            not (len(image.shape) != 3 
                or image.shape[-1] != 3)

def path_exists(path=None):
    if path and os.path.exists(path):
        return True
    return False
