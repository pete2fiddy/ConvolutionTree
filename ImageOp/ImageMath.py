import numpy

def set_images_to_binary(image, threshold):
    out_images = (image > threshold)
    out_images = out_images.astype(float)
    return out_images
    #out_images -= 0.5
    #return out_images * 2
'''
def normalize_images_to_range(images, range):
    out_images = images - ((range[0] - range[1])/2)
    '''
'''
def set_images_to_binary(images, threshold):
    out_images = numpy.zeros(images.shape)
    for i in range(0, images.shape[0]):
        out_images[i] = set_image_to_binary(images[i], threshold)
    return out_images
'''