from pathlib import Path
from scipy.misc import imread
import numpy as np

from utils import Utils
from omniglot_service import OmniglotService

class OmniglotLoader:
    
    def __init__(self, path='./data', train_folder='train_set', test_folder='test_set'):
        self.__path = Path(path)
        self.__train_folder = self.__path.joinpath(train_folder)
        self.__test_folder = self.__path.joinpath(test_folder)
        self.__train_file = self.__path.joinpath('train.pickle')
        self.__test_file = self.__path.joinpath('test.pickle')

        self.__data_service = OmniglotService(self.__path)
    
    def load_data(self):
        train_data = self.__load_data(self.__train_file, self.__train_folder)
        test_data = self.__load_data(self.__test_file, self.__test_folder, data_type='test')

        return (train_data, test_data)

    def __load_data(self, file, folder, data_type='train'):
        if file.exists():
            print(f'Loading data from {file}')
            data = Utils.read_data(str(file))
        else:
            if not folder.exists():
                self.__data_service.get_data_type(folder.name, data_type)
            data = self.__load_alphabets(folder)
            print(f'Saving data to {file}')
            Utils.save_data(str(file), data)

        return data
    
    def __load_alphabets(self, alphabets_folder):
        print(f'Loading data from {alphabets_folder}')
        alphabet_character_paths = self.__get_alphabet_character_paths(alphabets_folder)
        class_images = self.__get_class_images(alphabets_folder, alphabet_character_paths)

        return np.expand_dims(class_images, axis=class_images.ndim), alphabet_character_paths

    def __get_alphabet_character_paths(self, alphabets_folder):
        alphabet_character_paths = [
            alphabet_folder.name+'/'+character_folder.name
            for alphabet_folder in alphabets_folder.iterdir() if alphabet_folder.is_dir()
            for character_folder in alphabet_folder.iterdir() if character_folder.is_dir()
        ]

        return np.array(sorted(alphabet_character_paths))

    def __get_class_images(self, alphabets_folder, alphabet_character_paths):
        classes = [
            self.__get_images(alphabets_folder, alphabet_character_path)
            for alphabet_character_path in alphabet_character_paths
        ]

        return np.array(classes)

    def __get_images(self, parent_folder, alphabet_character_path):
        path = parent_folder.joinpath(alphabet_character_path)
        images = [
            imread(image_file) 
            for image_file in path.iterdir() if image_file.is_file()
        ]

        return images