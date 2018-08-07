from pathlib import Path
import numpy as np
from sklearn.utils import shuffle
from keras import backend as K

from utils import Utils

class OneShotRunner:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        
        self.__path = Path('./model/')
        if not self.__path.exists():
            self.__path.mkdir(parents=True, exist_ok=True)

        self.__model_weights_file = self.__path.joinpath('model_weights.h5')
        self.__training_data_file = self.__path.joinpath('training_data.pickle')
        self.__predictions_data_file = self.__path.joinpath('predictions_data.pickle')
        
        self.__training_loss = []
        self.__training_accuracy = []

        self.__validation_loss = []
        self.__validation_accuracy = []
        
        self.__report_every = 100
        self.__evaluate_every = 1000
        self.__save_every = 1000

    def __preload_weights(self):
        if self.__model_weights_file.exists():
            print('Preloading model weights')
            self.model.load_weights(str(self.__model_weights_file))

    def __preload_data(self):
        if self.__training_data_file.exists():
            print('Preloading training data')
            data = Utils.read_data(str(self.__training_data_file))
            self.__training_loss, self.__training_accuracy, self.__validation_loss, self.__validation_accuracy = data
    
    def train(self, number_way=20, number_iterations=10000, number_validations=50, preload_state=False):
        if preload_state:
            self.__preload_weights()
            self.__preload_data()

        print(f'Start training for {number_iterations} iterations with {number_validations} validations per each one-shot {number_way}-way task')
        for iteration in range(1, number_iterations+1):
            model_input, labels = self.get_train_batch()
            loss, accuracy = self.model.train_on_batch(model_input, labels)
            accuracy *= 100.
            if iteration % self.__report_every == 0:
                self.__training_loss.append(loss)
                self.__training_accuracy.append(accuracy)
                print(f'iteration {iteration}, loss = {loss:.2f}, accuracy = {accuracy:.2f}')

            if iteration % self.__evaluate_every == 0:
                loss, accuracy = self.__evaluate_one_shot(number_way, number_validations)
                self.__validation_loss.append(loss)
                self.__validation_accuracy.append(accuracy)
                print(f'evaluation at iteration {iteration}, loss = {loss:.2f}, accuracy = {accuracy:.2f}')

            if iteration % self.__save_every == 0:
                print(f'Saving training data at iteration {iteration}')
                data = self.__training_loss, self.__training_accuracy, self.__validation_loss, self.__validation_accuracy
                Utils.save_data(str(self.__training_data_file), data)
        
        print(f'Saving model weights')
        self.model.save_weights(self.__model_weights_file)
        
    def predict(self, number_way=20, number_iterations=100, number_validations=50, preload_state=False):
        if preload_state:
            self.__preload_weights()

        print(f'Start predictions for {number_iterations} iterations with {number_validations} validations per each one-shot {number_way}-way task')
        train_accuracy = []
        val_accuracy = []
        test_accuracy = []
        predict_every = 10
        for iteration in range(1, number_iterations+1):
            train_accuracy.append(self.__test_one_shot(number_way, number_validations, data_type='train'))
            val_accuracy.append(self.__test_one_shot(number_way, number_validations, data_type='val'))
            test_accuracy.append(self.__test_one_shot(number_way, number_validations))
            if iteration % predict_every == 0:
                print(f'Predictions at iteration {iteration} finished')
        data = train_accuracy, val_accuracy, test_accuracy
        print(f'Saving prediction data to {self.__predictions_data_file}')
        Utils.save_data(str(self.__predictions_data_file), data)

    def get_train_batch(self, batch_size=32):       
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
        left_encoder_input, rigth_encoder_input, labels = shuffle(left_encoder_input, rigth_encoder_input, labels)
        return [np.array(left_encoder_input), np.array(rigth_encoder_input)], labels

    def __evaluate_one_shot(self, number_way, number_validations, data_type='val'):
        accuracy = 0
        loss = 0
        for _ in range(number_validations):
            model_input, labels = self.get_one_shot_batch(number_way, data_type)
            labels_hat = self.model.predict_on_batch(model_input)
            y_true = K.variable(labels)
            y_pred = K.variable(labels_hat.squeeze())
            loss += K.eval(self.model.loss_functions[0](y_true, y_pred))
            correct = np.argmax(labels_hat)==np.argmax(labels)
            accuracy += int(correct)

        accuracy /= number_validations
        loss /= number_validations
        return loss, accuracy*100.

    def __test_one_shot(self, number_way, number_validations, data_type='test'):
        accuracy = 0
        for _ in range(number_validations):
            model_input, labels = self.get_one_shot_batch(number_way, data_type)
            labels_hat = self.model.predict_on_batch(model_input)
            correct = np.argmax(labels_hat)==np.argmax(labels)
            accuracy += int(correct)

        accuracy /= number_validations
        return accuracy*100.

    def get_one_shot_batch(self, batch_size, data_type):
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
            _, second_image = self.dataset.get_image_pair(class_value, data_type)
            left_encoder_input.append(first_image)
            rigth_encoder_input.append(second_image)

        left_encoder_input, rigth_encoder_input, labels = shuffle(left_encoder_input, rigth_encoder_input, labels)
        return [np.array(left_encoder_input), np.array(rigth_encoder_input)], labels