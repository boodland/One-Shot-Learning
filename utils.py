import pickle

class Utils:

    @staticmethod
    def save_data(file, data):
        with open(file, "wb") as f:
	        pickle.dump(data, f)
    
    @staticmethod
    def read_data(file):
        with open(file, "rb") as f:
            data = pickle.load(f)
        return data