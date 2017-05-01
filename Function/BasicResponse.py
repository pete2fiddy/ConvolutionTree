import numpy

def get_magnitude_of_response(original_image, conv_image):
    sub_arr = numpy.abs(conv_image - original_image)
    sub_arr = sub_arr.flatten()
    return sub_arr.sum()