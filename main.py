import os
import numpy as np
from PIL import Image

PATH = os.path.dirname(os.path.abspath(__file__))
FOLDER = '/chars74k-lite'

FULL_PATH = PATH + FOLDER

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
    images[letter] = letter_dict


