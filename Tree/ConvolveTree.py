import numpy
from scipy.ndimage.filters import convolve
from math import log
from PIL import Image
#from scipy.integrate import cumtrapz

class ConvolveTree(object):
    '''takes numpy array of binary numpy images'''
    def __init__(self, images, classifications, tree_depth, response_func):
        self.images = images
        self.num_classifications = numpy.unique(classifications).shape[0]
        self.response_func = response_func
        self.classifications = classifications
        self.init_tree(tree_depth)

    def init_tree(self, tree_depth):
        first_fork = Fork(self.images, self.classifications, self.response_func, self.num_classifications)
        self.forks = [[0 for i in range(0, 2**j)] for j in range(0, tree_depth)]
        self.forks[0][0] = first_fork
        for i in range(1, tree_depth):
            for j in range(0, len(self.forks[i])):
                '''if the index is odd, it is from the previous's fork negative, if not, it is from the positive split'''
                is_pos = (j%2 == 0)
                connected_fork = self.forks[i-1][int(j/2)]
                this_fork_data = None
                if not connected_fork.get_if_is_dead():
                    if is_pos:
                        this_fork_data = connected_fork.get_pos_splits()
                    else:
                        this_fork_data = connected_fork.get_neg_splits()
                    input_images = numpy.zeros((this_fork_data.shape[0], self.images.shape[1], self.images.shape[2]))
                else:
                    this_fork_data = numpy.array([])
                    input_images = numpy.zeros((this_fork_data.shape[0], self.images.shape[1], self.images.shape[2]))
                input_classifications = numpy.zeros((this_fork_data.shape[0]), dtype = int)
                for k in range(0, input_images.shape[0]):
                    input_images[k] = this_fork_data[k][1]
                    input_classifications[k] = int(this_fork_data[k][2])
                
                    
                append_fork = Fork(input_images, input_classifications, self.response_func, self.num_classifications)
                self.forks[i][j] = append_fork
                print("fork created at depth: ", i, ", with index: ", j)
                
    def predict(self, image):
        parent_fork_index = 0
        parent_response = self.forks[0][0].get_if_image_is_pos(image)
        parent_convolve = convolve(image, self.forks[0][parent_fork_index].get_fork_kernel())
        for depth_index in range(1, len(self.forks)):
            child_fork_index = parent_fork_index*2
            if not parent_response:
                child_fork_index += 1
            
            if self.forks[depth_index][child_fork_index].get_if_is_dead():
                if parent_response:
                    return self.forks[depth_index-1][parent_fork_index].get_classification_distributions()[0][1]
                return self.forks[depth_index-1][parent_fork_index].get_classification_distributions()[1][1]
                
            
            child_response = self.forks[depth_index][child_fork_index].get_if_image_is_pos(parent_convolve)
            
            parent_convolve = convolve(parent_convolve, self.forks[depth_index][child_fork_index].get_fork_kernel())
            parent_fork_index = child_fork_index
            parent_response = child_response
            
            
        if parent_response:
            return self.forks[len(self.forks)-1][parent_fork_index].get_classification_distributions()[0][1]
        return self.forks[len(self.forks)-1][parent_fork_index].get_classification_distributions()[1][1]
            
                

        
    def __str__(self):
        out_str = ""
        for i in range(0, len(self.forks)):
            out_str += "depth " + str(i) + ": "
            for j in range(0, len(self.forks[i])):
                out_str +=  str(self.forks[i][j]) 
                if j != len(self.forks[i]) - 1:
                    out_str += ", "
            out_str += "\n"
        return out_str
    
    def get_end_classification_distributions(self):
        cls_distributions = []
        for i in range(0, len(self.forks[len(self.forks)-1])):
            cls_distributions.append(self.forks[len(self.forks)-1][i].get_classification_distributions())
        return cls_distributions
    
    def get_end_max_classification_percentages(self):
        percents = []
        for i in range(0, len(self.forks[len(self.forks)-1])):
            percents.append(self.forks[len(self.forks)-1][i].get_max_classification_probability())
        return percents
    
class Fork(object):
    
    VAL_STEP = 0.2
    MAX_VAL = 1
    def __init__(self, fork_images, fork_classifications, response_func, num_classifications):
        self.response_func = response_func
        self.num_classifications = num_classifications
        self.fork_classifications = fork_classifications
        self.fork_images = fork_images
        self.set_if_dead()
        if not self.is_dead:
            self.init_fork_kernel()
        
    
    def set_if_dead(self):
        self.is_dead = (self.fork_images.shape[0] <= 1)
    
    def init_fork_kernel(self, kernel_size = 3, kernel_depth = 5):
        kernel = numpy.zeros((kernel_size, kernel_size))
        kernel_edited_mat = numpy.zeros((kernel_size, kernel_size), dtype = bool)
        for depth_index in range(0, kernel_depth):
            min_entropy_kernel_index = (0,0)
            min_entropy = None
            min_entropy_val = None
            for i in range(0, kernel_size):
                for j in range(0, kernel_size):
                    if not kernel_edited_mat[i,j]:
                        val = 0
                        while val <= Fork.MAX_VAL:
                            kernel_copy = kernel.copy()
                            kernel_copy[i,j] = val
                            self.fork_kernel = kernel_copy
                            self.run_set()
                            iter_entropy = self.total_entropy
                            if min_entropy == None or iter_entropy < min_entropy:
                                min_entropy = iter_entropy
                                min_entropy_kernel_index = (i,j)
                                min_entropy_val = val
                            val += Fork.VAL_STEP
            kernel[min_entropy_kernel_index[0], min_entropy_kernel_index[1]] = min_entropy_val
            kernel_edited_mat[min_entropy_kernel_index[0], min_entropy_kernel_index[1]] = True
        self.fork_kernel = kernel
        self.run_set()

    def run_set(self):
        self.init_kernel_response_images()
        self.init_magnitude_response_arr()
        self.split()     
        self.init_total_entropy()
        
    
    
    def init_kernel_response_images(self):
        self.kernel_responses = numpy.zeros(self.fork_images.shape)
        for i in range(0, self.fork_images.shape[0]):
            kernel_image = convolve(self.fork_images[i], self.fork_kernel)
            self.kernel_responses[i] = kernel_image
        self.kernel_responses = numpy.asarray(self.kernel_responses)
        
    def init_magnitude_response_arr(self):
        self.response_mags = numpy.zeros((self.kernel_responses.shape[0]))
        for i in range(0, self.kernel_responses.shape[0]):
            response_magnitude = self.response_func.get_magnitude_of_response(self.fork_images[i], self.kernel_responses[i])
            self.response_mags[i] = response_magnitude
            
    '''splits the images into two categories based on the magnitude of the response to the fork kernel.
    In the future, the best split should minimize the entropy of the fork set by trying values between
    all consecutive response magnitudes, but for now it just uses the mean'''
            
    '''keep in mind when choosing a different split value other than the mean that, if you sort the responses,
    you can easily segment positive and negative responses by just chopping the list at the value you just chose
    in between two responses'''
    
    '''positive and negative splits are stored in their respective "pos_splits" and "neg_splits" arrays
    as tuples, where:
    [0] is the original image
    [1] is the kernel response image
    [2] is the classification
    [3] is the magnitude of the response to the fork kernel
    '''
    def split(self):
        self.split_val = numpy.average(self.response_mags)
        self.pos_splits = []
        self.neg_splits = []
        for i in range(0, self.response_mags.shape[0]):
            append_tuple = (self.fork_images[i], self.kernel_responses[i], self.fork_classifications[i], self.response_mags[i])
            if self.response_mags[i] > self.split_val:
                self.pos_splits.append(append_tuple)
            else:
                self.neg_splits.append(append_tuple)
        self.pos_splits = numpy.asarray(self.pos_splits)
        self.neg_splits = numpy.asarray(self.neg_splits)
        
    def init_total_entropy(self):
        num_pos_classifications = numpy.zeros((self.num_classifications))
        num_neg_classifications = num_pos_classifications.copy()
        for i in range(0, self.pos_splits.shape[0]):
            class_index = self.pos_splits[i][2]
            num_pos_classifications[class_index] += 1.0
        for i in range(0, self.neg_splits.shape[0]):
            class_index = self.neg_splits[i][2]
            num_neg_classifications[class_index] += 1.0
            
        self.pos_classification_probs = num_pos_classifications/numpy.sum(num_pos_classifications)
        self.neg_classification_probs = num_neg_classifications/numpy.sum(num_neg_classifications)
        
        entropy_positive = self.get_entropy(self.pos_classification_probs)
        entropy_negative = self.get_entropy(self.neg_classification_probs)
        
        proportion_pos = float(self.pos_splits.shape[0])/float(self.fork_images.shape[0])
        proportion_neg = float(self.neg_splits.shape[0])/float(self.fork_images.shape[0])
        '''may have made a mistake here'''
        self.total_entropy = proportion_pos * entropy_positive + proportion_neg * entropy_negative
        
    def get_entropy(self, prob_dist):
        sum = 0
        for i in range(0, prob_dist.shape[0]):
            if prob_dist[i] != 0:
                add_num = prob_dist[i] * log(prob_dist[i], 2.0)
                sum += add_num
        return -1 * sum
        
    def get_pos_splits(self):
        return self.pos_splits

    def get_neg_splits(self):
        return self.neg_splits
        
    def __str__(self):
        if not self.is_dead:
            return "Fork with : " + str(self.total_entropy) + " entropy, and " + str(numpy.average(self.response_mags)) + " average response magnitude"
        return "DEAD FORK"
        
    def get_classification_distributions(self):
        if not self.is_dead:
            return (("positive distribution: ", self.pos_classification_probs), ("negative distribution: ", self.neg_classification_probs))    
        return (("DEAD FORK"), ("DEAD FORK"))

    def get_max_classification_probability(self):
        if not self.is_dead:
            return (("max pos %: ", (numpy.amax(self.pos_classification_probs)*10000)//100), ("max neg %:", (numpy.amax(self.neg_classification_probs)*10000)//100))
        return (("DEAD FORK"), ("DEAD FORK"))
    
    def get_if_is_dead(self):
        return self.is_dead
    
    def get_if_image_is_pos(self, image):
        kernel_response = convolve(image, self.fork_kernel)
        magnitude_response = self.response_func.get_magnitude_of_response(image, kernel_response)
        return magnitude_response > self.split_val
    
    def get_fork_kernel(self):
        return self.fork_kernel
    
    