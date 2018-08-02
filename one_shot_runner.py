from pathlib import Path
import numpy as np
from sklearn.utils import shuffle
import pickle

class OneShotRunner():
    def __init__(self, dataset, model, preload_data=True):
        self.dataset = dataset
        self.model = model

        self.__number_ways = 20 
        
        self.__path = Path('./model/')
        if not self.__path.exists():
            self.__path.mkdir(parents=True, exist_ok=True)

        self.__best_accuracy_weights_file = self.__path.joinpath('best_accuracy_weights.h5')
        self.__training_data_file = self.__path.joinpath('training_data.pickle')
        
        self.__training_loss = []
        self.__training_accuracy = []
        
        self.__best_accuracy = 0
        self.__evaluate_every = 1000
        self.__loss_every = 100
        self.__store_every = 1000

        if preload_data:
            self.__preload_data()

    def __preload_data(self):
        print('Preloading runner data')
        if self.__best_accuracy_weights_file.exists():
            self.model.load_weights(str(self.__best_accuracy_weights_file))
        
        if self.__training_data_file.exists():
            self.__read_data()
    
    def train(self, number_iterations=10000, num_validation=500):
        print(f'Start training for {number_iterations} iterations with {num_validation} validations per evaluation')
        for iteration in range(1, number_iterations):
            model_input, labels = self.__get_train_batch()
            loss = self.model.train_on_batch(model_input, labels)

            if iteration % self.__loss_every == 0:
                self.__training_loss.append(loss)
                print(f'iteration {iteration}, training loss: {loss:.2f}')

            if iteration % self.__evaluate_every == 0:
                accuracy = self.__evaluate(iteration, num_validation)
                self.__training_accuracy.append(accuracy)
                print(f'Accuracy at iteration {iteration} = {accuracy}')

            if iteration % self.__store_every == 0:
                self.__save_data()
                print(f'Saving training data at iteration {iteration}')
        
        print(f'Data after training: loss = {loss}, best accuracy {self.__best_accuracy:.2f}')
        
    def __evaluate(self, iteration, num_validation):
        accuracy = self.__test_one_shot(self.__number_ways, num_validation)
        if accuracy > self.__best_accuracy:
            print(f'Saving model weights with best accuracy of {accuracy:.2f}')
            self.model.save_weights(self.__best_accuracy_weights_file)
            self.__best_accuracy = accuracy

        return accuracy

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

    def __save_data(self):
        data = self.__training_loss, self.__training_accuracy, self.__best_accuracy
        with open(str(self.__training_data_file), "wb") as f:
	        pickle.dump(data, f)

    def __read_data(self):
        with open(str(self.__training_data_file), "rb") as f:
            data = pickle.load(f)
        self.__training_loss, self.__training_accuracy, self.__best_accuracy = data