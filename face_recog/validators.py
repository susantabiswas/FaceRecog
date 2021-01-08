def is_valid_img(image):
    return not (len(image.shape) != 3 
                or image.shape[-1] != 3)
