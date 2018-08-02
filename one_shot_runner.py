from pathlib import Path
import numpy as np
from sklearn.utils import shuffle
from utils import Utils

class OneShotRunner():
    def __init__(self, dataset, model, preload_data=True):
        self.dataset = dataset
        self.model = model
        
        self.__path = Path('./model/')
        if not self.__path.exists():
            self.__path.mkdir(parents=True, exist_ok=True)

        self.__best_accuracy_weights_file = self.__path.joinpath('best_accuracy_weights.h5')
        self.__training_data_file = self.__path.joinpath('training_data.pickle')
        self.__predictions_data_file = self.__path.joinpath('predictions_data.pickle')
        
        self.__training_loss = []
        self.__training_accuracy = []
        
        self.__best_accuracy = 0
        self.__evaluate_every = 1000
        self.__loss_every = 100
        self.__store_every = 1000

        if preload_data:
            self.__preload_data()

    def __preload_data(self):
        if self.__best_accuracy_weights_file.exists():
            print('Preloading best accuracy model weights')
            self.model.load_weights(str(self.__best_accuracy_weights_file))
        
        if self.__training_data_file.exists():
            print('Preloading training data')
            data = Utils.read_data(str(self.__training_data_file))
            self.__training_loss, self.__training_accuracy, self.__best_accuracy = data
    
    def train(self, number_ways=20, number_iterations=10000, number_validations=500):
        print(f'Start training for {number_iterations} iterations with {number_validations} validations per each {number_ways} ways evaluation')
        for iteration in range(1, number_iterations+1):
            model_input, labels = self.__get_train_batch()
            loss = self.model.train_on_batch(model_input, labels)

            if iteration % self.__loss_every == 0:
                self.__training_loss.append(loss)
                print(f'iteration {iteration}, loss = {loss:.2f}')

            if iteration % self.__evaluate_every == 0:
                accuracy = self.__evaluate(number_ways, iteration, number_validations)
                self.__training_accuracy.append(accuracy)
                print(f'iteration {iteration}, accuracy = {accuracy:.2f}')

            if iteration % self.__store_every == 0:
                print(f'Saving training data at iteration {iteration}')
                data = self.__training_loss, self.__training_accuracy, self.__best_accuracy
                Utils.save_data(str(self.__training_data_file), data)
        
        print(f'Data after training: loss = {loss}, best accuracy = {self.__best_accuracy:.2f}')
        
    def predict(self, number_ways=20, number_iterations=100, number_validations=50):
        print(f'Start predictions for {number_iterations} iterations with {number_validations} validations per each {number_ways} ways prediction')
        train_accuracy = []
        val_accuracy = []
        test_accuracy = []
        predict_every = 10
        for iteration in range(1, number_iterations+1):
            train_accuracy.append(self.__test_one_shot(number_ways, number_validations, data_type='train'))
            val_accuracy.append(self.__test_one_shot(number_ways, number_validations))
            test_accuracy.append(self.__test_one_shot(number_ways, number_validations, data_type='test'))
            if iteration % predict_every == 0:
                print(f'Predictions at iteration {iteration} finished')
        data = train_accuracy, val_accuracy, test_accuracy
        Utils.save_data(str(self.__predictions_data_file), data)

    def __evaluate(self, number_ways, iteration, number_validations):
        accuracy = self.__test_one_shot(number_ways, number_validations)
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

    def __test_one_shot(self, number_ways, number_validations, data_type='val'):
        accuracy = 0
        for _ in range(number_validations):
            model_input, labels = self.__get_one_shot_batch(number_ways, data_type)
            labels_hat = self.model.predict_on_batch(model_input)
            correct = np.argmax(labels_hat)==np.argmax(labels)
            accuracy += int(correct)

        accuracy /= number_validations
    
        return accuracy*100.

    def __get_one_shot_batch(self, batch_size, data_type):
        left_encoder_input = []
        rigth_encoder_input = []
        labels = np.zeros((batch_size,))

        classes_to_sample = self.dataset.get_data_classes(batch_size, data_type)
        true_class = classes_to_sample[0]

        first_image, second_image = self.dataset.get_image_pair(true_class, data_type, same_class=True)
        left_encoder_input.append(first_image)
        rigth_encoder_input.append(second_image)
        labels[0] = 1

        for class_value in classes_to_sample[1:]:
            _, second_image = self.dataset.get_image_pair(class_value, data_type=data_type)
            left_encoder_input.append(first_image)
            rigth_encoder_input.append(second_image)

        left_encoder_input, rigth_encoder_input, labels = shuffle(left_encoder_input, rigth_encoder_input, labels)
        return [np.array(left_encoder_input), np.array(rigth_encoder_input)], labels