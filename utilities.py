from copy import deepcopy as dcopy
import numpy as np
import pickle
# Additional necessary functions (from config.py)
class dict2class(object):
    """
    Converts dictionary into class object
    Dict key,value pairs become attributes
    """
    def __init__(self, dict):
        for key, val in dict.items():
            setattr(self, key, val)
                                
def is_odd(num):
   return num % 2 != 0


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

# Package for output
def convertSave(initDict,modelStruct):
    outPack = dict()
    outPack['Task'] = dcopy(initDict)
    outPack['Model'] = dcopy(modelStruct)
    return(outPack)
