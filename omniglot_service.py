from pathlib import Path
import urllib.request
import zipfile

class OmniglotService:
    
    def __init__(self, path='./data/'):
        self.__path = Path(path)
        if not self.__path.exists():
            self.__path.mkdir(parents=True, exist_ok=True)
        
        self.__root_url = "https://github.com/brendenlake/omniglot/raw/master/python/"
        self.__train_tag = "images_background"
        self.__test_tag = "images_evaluation"
        self.__filename_extension = '.zip'
        
    def get_data(self):
        self.get_data_type('train_set')
        self.get_data_type('test_set', data_type='test')

    def get_data_type(self, folder_name, data_type='train'):
        to_folder = self.__path.joinpath(folder_name)
        if not to_folder.exists():
            tag = self.__train_tag if (data_type == 'train') else self.__test_tag
            self.__get_data(tag, to_folder)
            
    def __get_data(self, data_type, to_folder):
        data_url = self.__root_url + data_type + self.__filename_extension
        filename = self.__path.joinpath(data_type + self.__filename_extension)
        self.__download_data(data_url, str(filename))
        self.__unzip_data(filename)
        self.__rename_data_folder(data_type, to_folder)
            
    def __download_data(self, data_url, filename):
        print(f'Downloading data from {data_url}')
        urllib.request.urlretrieve(data_url, filename)
        
    def __unzip_data(self, filename):
        print(f'Unziping {filename} to {self.__path}')
        with zipfile.ZipFile(str(filename),"r") as zip_ref:
            zip_ref.extractall(self.__path)
        filename.unlink()
            
    def __rename_data_folder(self, data_type, target_folder):
        origin = self.__path.joinpath(data_type)
        print(f'Rename {origin} to {target_folder}')
        origin.rename(target_folder)