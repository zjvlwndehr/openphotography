import cv2
import numpy as np
from pathlib import Path

class Train():
    def __init__(self, path : Path, ext : str = 'jpg') -> None:
        self.path = path
        self.ext = ext
        _exists(path, ext)
        self.x_train = [i for i in path.glob(f'*.{ext}')]
        self.y_train = [1.0 for i in range(len(self.x_train))]
        self.train = list(zip(self.x_train, self.y_train))

    def get_images(self, count : int = 8, size : tuple = None, crop : tuple = None, rotate : int = None, flip : int = None) -> list:
        ret = []
        for i in range(count):
            index = np.random.randint(0, len(self.x_train))
            img = cv2.imread(str(self.train[index][0]))
            if crop is not None:
                img = _crop(img, *crop)
            if rotate is not None:
                img = _rotate(img, rotate)
            if flip is not None:
                img = _flip(img, flip)
            if size is not None or type(size) is tuple:
                if size != (img.shape[0], img.shape[1], img.shape[2]):
                    img = _resize(img, size)
            ret.append(img)
        return ret
    
class Test():
    def __init__(self, path : Path, ext : str = 'jpg') -> None:
        self.path = path
        self.ext = ext
        _exists(path, ext)
        self.x_test = []
        self.y_test = []        

def _exists(path : Path, ext : str = 'jpg'):
    _path_exists(path)
    for file in path.glob(f'*.{ext}'):
        _file_exists(file)

def _resize(img, size : tuple):
    return cv2.resize(img, size)

def _crop(img, x : int, y : int, w : int, h : int):
    return img[y:y+h, x:x+w]

def _rotate(img, angle : int):
    return cv2.rotate(img, angle)

def _flip(img, flipCode : int):
    return cv2.flip(img, flipCode)

def _path_exists(path : Path):
    if not path.exists():
        raise FileNotFoundError(f'[ERROR]: {path} does not exist.')

def _file_exists(path : Path):
    if not path.is_file():
        raise FileNotFoundError(f'[ERROR]: {path} does not exist.')

if __name__ == '__main__':
    assert False, '[INFO]: This is a module. To use it, import it.'