from sklearn.datasets import fetch_mldata, load_digits
from Tree.ConvolveTree import ConvolveTree
import ImageOp.ImageMath as ImageMath
from PIL import Image
import Function.BasicResponse as BasicResponse
import numpy

digits = load_digits()#fetch_mldata('MNIST original', data_home = custom_data_home)
print("digits shape: ", digits.data.shape)
num_training = 1000#digits.data.shape[0]
images = ImageMath.set_images_to_binary(digits.data.reshape((digits.data.shape[0], 8, 8)), 4)
training_images = images[0:num_training]
training_targets = digits.target[0:num_training]

testing_images = images[num_training: images.shape[0]]
testing_targets = digits.target[num_training:digits.target.shape[0]]

tree = ConvolveTree(training_images, training_targets, 9, BasicResponse)
print("tree: \n", tree)
print("max class distributions: ", tree.get_end_max_classification_percentages())

num_correct = 0
num_possible = 0
for i in range(0, testing_images.shape[0]):
    prediction_distribution = tree.predict(testing_images[i])
    predicted_class = numpy.argmax(prediction_distribution)
    if not prediction_distribution[0] == "DEAD FORK" or not numpy.isnan(prediction_distribution[predicted_class]):
        target = testing_targets[i]
        if predicted_class == target:
            num_correct += 1
        num_possible += 1
print("num possible was: ", num_possible)
print("percent correct: ", 100*float(num_correct)/float(num_possible))