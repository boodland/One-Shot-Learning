from pathlib import Path
import numpy as np
from sklearn.utils import shuffle

class OneShotRunner():
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

        self.__number_ways = 20
    
    def train(self, number_iterations=10000, num_validation=500):
        test_every = 1000 # interval for evaluating on one-shot tasks
        loss_every=100 # interval for printing loss (iterations)
        #weights_path = os.path.join(PATH, "weights")
        print("training")
        for i in range(number_iterations):
            model_input, labels = self.__get_train_batch()
            loss = self.model.train_on_batch(model_input, labels)

            if i % loss_every == 0:
                print("iteration {}, training loss: {:.2f},".format(i,loss))

            if i % test_every == 0:
                print("evaluating")
                accuracy = self.__test_one_shot(self.__number_ways, num_validation)
                print(f'Accuracy at iteration {i} = {accuracy}')
        
        print("training loss after training: {:.2f},".format(loss))
        
    def __get_train_batch(self, batch_size=32):       
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

    def __test_one_shot(self, num_ways, num_validation):
        accuracy = 0
        for _ in range(num_validation):
            model_input, labels = self.__get_one_shot_batch(num_ways)
            labels_hat = self.model.predict_on_batch(model_input)
            correct = np.argmax(labels_hat)==np.argmax(labels)
            accuracy += int(correct)

        accuracy /= num_validation
    
        return accuracy*100.

    def __get_one_shot_batch(self, batch_size=20):
        left_encoder_input = []
        rigth_encoder_input = []
        labels = np.zeros((batch_size,))

        classes_to_sample = self.dataset.get_data_classes(batch_size, data_type='test')
        true_class = classes_to_sample[0]

        first_image, second_image = self.dataset.get_image_pair(true_class, same_class=True, data_type='test')
        left_encoder_input.append(first_image)
        rigth_encoder_input.append(second_image)
        labels[0] = 1

        for class_value in classes_to_sample[1:]:
            _, second_image = self.dataset.get_image_pair(class_value, data_type='test')
            left_encoder_input.append(first_image)
            rigth_encoder_input.append(second_image)

        left_encoder_input, rigth_encoder_input, labels = shuffle(left_encoder_input, rigth_encoder_input, labels)
        return [np.array(left_encoder_input), np.array(rigth_encoder_input)], labels