from omniglot_loader import OmniglotLoader

import numpy as np

class OmniglotDataset:
    
    def __init__(self, path="./data/"):
        
        self.__path = path
        self.__train_folder = "train_alphabets"
        self.__test_folder = "test_alphabets"
        
        self.__data_loader = OmniglotLoader(self.__path, self.__train_folder, self.__test_folder)
        
        self.train_data = ()
        self.val_data = ()
        self.test_data = ()
        self.data_shape = ()
    
    def load(self):
        train_val_data, self.test_data = self.__data_loader.load_data()
        _, _, height, width, channel = train_val_data[0].shape
        start_index_val = train_val_data[1].index('Latin/character01')
        self.train_data = (train_val_data[0][:start_index_val], train_val_data[1][:start_index_val])
        self.val_data =  (train_val_data[0][start_index_val:], train_val_data[1][start_index_val:])
        self.data_shape = (height, width, channel)

    def get_data_classes(self, num_classes, data_type='train'):
        data = self.__get_data_type(data_type)
        classes = np.random.choice(data.shape[0], size=(num_classes,), replace=False)
        
        return classes

    def get_image_pair(self, class_value, data_type='train', same_class=False):
        data = self.__get_data_type(data_type)
        num_classes, num_images, _, _, _ = data.shape
        image_indices = np.random.choice(num_images, size=2, replace=(not same_class))
        
        first_image = data[class_value, image_indices[0]]
        
        second_class = class_value if same_class else self.__get_different_value(class_value, num_classes)
        second_image = data[second_class, image_indices[1]]

        return first_image, second_image

    def __get_different_value(self, value, max_values):
        values = list(range(max_values))
        values.remove(value)
        different_value = np.random.choice(values)
        
        return different_value

    def __get_data_type(self, data_type):
        if (data_type == 'train'): 
            return self.train_data[0]
        elif (data_type == 'val'): 
            return self.val_data[0]
        else:
            return self.test_data[0]

