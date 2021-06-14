import pickle

def save_obj(obj, file ):
    """save an object as .pkl file

    obj - obj to save
    name - name under which the object is to be saved 
    """
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file):
    """load an object

    name - name of the object to be loaded
    """
    with open(file, 'rb') as f:
        return pickle.load(f)