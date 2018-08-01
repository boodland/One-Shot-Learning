from pathlib import Path
import numpy as np
from sklearn.utils import shuffle

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
        labels = np.zeros((batch_size,))
        labels[half_batch_size:] = 1
        
        classes_to_sample = self.dataset.get_data_classes(batch_size)
        for class_ind, class_value in enumerate(classes_to_sample):
            is_same_class = (class_ind >= half_batch_size)
            first_image, second_image = self.dataset.get_image_pair(class_value, same_class=is_same_class)
            left_encoder_input.append(first_image)
            rigth_encoder_input.append(second_image)
            
        return [np.array(left_encoder_input), np.array(rigth_encoder_input)], labels

    def __get_one_shot_batch(self, batch_size=20):
        left_encoder_input = []
        rigth_encoder_input = []
        labels = np.zeros((batch_size,))

        classes_to_sample = self.dataset.get_data_classes(batch_size, data_type='test')
        true_class = classes_to_sample[0]

        first_image, second_image = self.dataset.get_image_pair(true_class, same_class=True)
        left_encoder_input.append(first_image)
        rigth_encoder_input.append(second_image)
        labels[0] = 1

        for class_value in classes_to_sample[1:]:
            _, second_image = self.dataset.get_image_pair(class_value)
            left_encoder_input.append(first_image)
            rigth_encoder_input.append(second_image)

        left_encoder_input, rigth_encoder_input, labels = shuffle(left_encoder_input, rigth_encoder_input, labels)
        return np.array(left_encoder_input), np.array(rigth_encoder_input), labels