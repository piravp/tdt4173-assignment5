import os
import numpy as np
from PIL import Image
from sklearn import preprocessing as pp
import random
from sklearn import svm

PATH = os.path.dirname(os.path.abspath(__file__))
FOLDER = '/chars74k-lite'


FULL_PATH = PATH + FOLDER

def load_images():
    images = dict()
    first = True
    for folder in os.walk(FULL_PATH):
        letter_path = folder[0]
        letter = letter_path[-1]
        if first:
            first = False
            continue
        idx = 0
        letter_dict = dict()
        for file in os.listdir(letter_path):
            if file.endswith(".jpg"):
                image = Image.open(letter_path + '/' + file)
                pix = np.array(image)
                letter_dict[idx] = pix
                idx += 1
        images[letter] = np.array(letter_dict.values())
    return images
            

def pre_scaling(images):
    for letter in images:
        for img_idx in range (0, images[letter].size):
            img = images[letter][img_idx]
            images[letter][img_idx] = pp.scale(img/255)
    return images

def divide_dataset(images, test_percentage):
    test_set = dict()
    for letter in images:
        letter_dict = dict()
        num_pics = len(images[letter])
        test_size = round(num_pics*test_percentage)
        idxs = random.sample(range(0,num_pics), test_size)
        for idx in idxs:
            letter_dict[idx] = images[letter][idx]
            del images[letter][idx]
        test_set[letter] = letter_dict
    return test_set

def svm_train(images):
    pass

images = load_images()
images = pre_scaling(images)
test_images = divide_dataset(images, 0.2)


        



