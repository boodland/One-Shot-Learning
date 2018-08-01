from pathlib import Path
from scipy.misc import imread
import numpy as np

class OmniglotLoader:
    
    def __init__(self, path, train_folder, test_folder):
        self.__path = Path(path)
        self.__train_folder = self.__path.joinpath(train_folder)
        self.__test_folder =self.__path.joinpath(test_folder)
    
    def load_data(self):
        train_data = self.__load_alphabets(self.__train_folder)
        test_data = self.__load_alphabets(self.__test_folder)

        return (train_data, test_data)
    
    def __load_alphabets(self, alphabets_folder):
        alphabet_character_paths = self.__get_alphabet_character_paths(alphabets_folder)
        class_images = self.__get_class_images(alphabets_folder, alphabet_character_paths)

        return np.expand_dims(class_images, axis=class_images.ndim), alphabet_character_paths

    def __get_alphabet_character_paths(self, alphabets_folder):
        alphabet_character_paths = [
            alphabet_folder.name+'/'+character_folder.name
            for alphabet_folder in alphabets_folder.iterdir() if alphabet_folder.is_dir()
            for character_folder in alphabet_folder.iterdir() if character_folder.is_dir()
        ]

        return sorted(alphabet_character_paths)

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
        