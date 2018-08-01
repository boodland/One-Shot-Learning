from pathlib import Path
import numpy as np

class OneShotRunner():
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

        self.__number_ways = 20
        self.__number_validations = 250
    
    def __get_batch(self, batch_size=32):
        
        half_batch_size = batch_size // 2
        left_encoder_input = []
        rigth_encoder_input = []
        targets = np.zeros((batch_size,))
        targets[half_batch_size:] = 1
        
        classes_to_sample = self.dataset.get_data_classes(batch_size)
        for class_ind, class_value in enumerate(classes_to_sample):
            same_class = (class_ind >= half_batch_size)
            first_image, second_image = self.dataset.get_image_pair(class_value, same_class=same_class)
            left_encoder_input.append(first_image)
            rigth_encoder_input.append(second_image)
            
        return left_encoder_input, rigth_encoder_input, targets