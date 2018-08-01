from omniglot_service import OmniglotService
from omniglot_loader import OmniglotLoader

import numpy as np

class OmniglotDataset:
    
    def __init__(self, path="./data/"):
        
        self.__path = path
        self.__train_folder = "train_alphabets"
        self.__test_folder = "test_alphabets"
        
        self.__data_service = OmniglotService(self.__path, self.__train_folder, self.__test_folder)
        self.__data_loader = OmniglotLoader(self.__path, self.__train_folder, self.__test_folder)
        
        self.train_data = ()
        self.test_data = ()
        self.input_shape = ()
    
    def load(self):
        self.__data_service.get_data()
        self.train_data, self.test_data = self.__data_loader.load_data()
        _, _, height, width = self.train_data[0].shape
        self.data_shape = (height, width, 1)

    def get_batch(self, batch_size=32):
        
        half_batch_size = batch_size // 2
        num_classes, _, height, width = self.train_data[0].shape
        classes_to_sample = np.random.choice(num_classes, size=(batch_size,), replace=False)
        pairs = [np.zeros((batch_size, height, width, 1)) for i in range(2)]
        targets = np.zeros((batch_size,))
        targets[half_batch_size:] = 1

        for batch_ind in range(0, half_batch_size):
            class_ind = classes_to_sample[batch_ind]
            first_pair, second_pair = self.__get_different_pair(class_ind)
            pairs[0][batch_ind,:,:,:] = first_pair
            pairs[1][batch_ind,:,:,:] = second_pair
            
        for batch_ind in range(half_batch_size, batch_size):
            class_ind = classes_to_sample[batch_ind]
            first_pair, second_pair = self.__get_same_pair(class_ind)
            pairs[0][batch_ind,:,:,:] = first_pair
            pairs[1][batch_ind,:,:,:] = second_pair
            
        return pairs, targets

    def __get_same_pair(self, class_ind):
        _, num_images, height, width = self.train_data[0].shape
        image_indices = list(range(0, num_images))
        image_ind = np.random.choice(image_indices)
        first_pair = self.train_data[0][class_ind, image_ind].reshape(height, width, 1)
        
        image_indices.remove(image_ind)
        image_ind = np.random.choice(image_indices)
        second_pair = self.train_data[0][class_ind, image_ind].reshape(height, width, 1)

        return first_pair, second_pair

    def __get_different_pair(self, class_ind):
        num_classes, num_images, height, width = self.train_data[0].shape
        image_ind = np.random.randint(0, num_images)
        first_pair = self.train_data[0][class_ind, image_ind].reshape(height, width, 1)
        
        classes_indices = list(range(0, num_classes))
        classes_indices.remove(class_ind)
        image_ind = np.random.randint(0, num_images)
        class_ind = np.random.choice(classes_indices)
        second_pair = self.train_data[0][class_ind, image_ind].reshape(height, width, 1)

        return first_pair, second_pair
