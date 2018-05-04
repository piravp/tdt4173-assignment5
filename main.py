import os
import numpy as np
from PIL import Image
from sklearn import preprocessing as pp
import random
from sklearn.svm import SVC
from collections import OrderedDict
import skimage.feature as skimage
from sklearn.neural_network import MLPClassifier
import matplotlib as plt

PATH = os.path.dirname(os.path.abspath(__file__))
FOLDER = '/chars74k-lite'


FULL_PATH = PATH + FOLDER

def load_images(flatten):
    images = OrderedDict()
    first = True
    for folder in os.walk(FULL_PATH):
        letter_path = folder[0]
        letter = letter_path[-1]
        if first:
            first = False
            continue
        idx = 0
        letters = [[]]
        for file in os.listdir(letter_path):
            if file.endswith(".jpg"):
                image = Image.open(letter_path + '/' + file)
                pix = np.array(image)
                if flatten:
                    pix = pix.flatten()
                letters.append(pix/255)
                idx += 1
        images[letter] = np.array(letters)[1:]
    return images
            

def pre_scaling(images):
    for letter in images:
        for img_idx in range (0, images[letter].size):
            img = images[letter][img_idx]
            images[letter][img_idx] = pp.scale(img)
    return images

def pre_hog(images):
    for letter in images:
        for idx in range(0,len(images[letter])):
            img = images[letter][idx]
            images[letter][idx] = skimage.hog(img, orientations = 10,pixels_per_cell=(5,5), cells_per_block=(1,1))
    return images

def divide_dataset(images, test_percentage):
    test_set = OrderedDict()
    for letter in images:
        letter_list = [[]]
        num_pics = len(images[letter])
        test_size = round(num_pics*test_percentage)
        idxs = random.sample(range(0,num_pics), test_size)
        for idx in idxs:
            letter_list.append(images[letter][idx])
        images[letter] = np.delete(images[letter],idxs)
        test_set[letter] = np.array(letter_list)[1:]
    return test_set

def svm_fit(x,y):
    clf = SVC(probability=True)
    clf.fit(x,y)
    return clf

def svm_predict(clf, test_x, test_y):
    correct = 0
    total = 0
    for i in range(0, len(test_x)):
        y_predicted = clf.predict([test_x[i]])
        if y_predicted == test_y[i]:
            correct += 1
        total += 1
    return correct/total
    
def create_labels(images):
    x = []
    y = []
    idx = 0
    for letter_list in images.values():
        for img in letter_list:
            y.append(idx)
            x.append(img)     
        idx += 1
    return x,y
    
def nn_fit(x,y):
    clf = MLPClassifier(solver='lbfgs', alpha=0.05, hidden_layer_sizes=(120, 80, 40), random_state=1, learning_rate_init = 0.2, learning_rate = 'adaptive')
    clf.fit(x,y)
    return clf

def nn_predict(clf, test_x , test_y):
    correct = 0
    total = 0
    for i in range(0, len(test_x)):
        y_predicted = clf.predict([test_x[i]])
        if y_predicted == test_y[i]:
            correct += 1
        total += 1
    return correct/total

def load_detection_image(filename):
    image = Image.open(filename)
    pix = np.array(image)
    return pix

def crop_image(image, sensitivity):
    return image
    
def detection(filename, clf, window_size, HOG, SCALING):
    print("Detecting...")
    threshold = 0.5
    image = load_detection_image(filename)
    image = crop_image(image, 3)
    h = image.shape[0]
    l = image.shape[1]
    pot_chars = []
    for row in range(0,h-window_size,4):
        for col in range(0,l-window_size,4):
            sub_img = image[np.ix_(range(row,row+window_size),range(col,col+window_size))]
            sub_img = sub_img/255
            if HOG:
                sub_img = skimage.hog(sub_img, orientations = 10,pixels_per_cell=(5,5), cells_per_block=(1,1))
            if SCALING:
                sub_img = pp.scale(sub_img)
            if not HOG:
                sub_img = sub_img.flatten()
            #Shape 1, 26
            res = clf.predict_proba(sub_img)
            res_max = 0
            idx = 0
            for i in range(0,26):
                if res[0][i] > res_max:
                    res_max = res[0][i]
                    idx = i
            if res_max > threshold:
                pot_chars.append((row,col,idx))
    return pot_chars
    
def show_chars(pot_chars, window_size):
    pass
    


def run(SVM, NN, SCALING, HOG, CLASSIFICATION, DETECTION, detection_filename):
    print("Loading...")
    images = load_images(not HOG)
    if HOG:
        images = pre_hog(images)
    if SCALING:
        images = pre_scaling(images)
    test_images = divide_dataset(images, 0.2)
    train_x, train_y = create_labels(images)
    test_x, test_y = create_labels(test_images)
    print("Fitting...")
    if SVM:
        clf = svm_fit(train_x, train_y)
        if CLASSIFICATION:
            result = svm_predict(clf, test_x, test_y)
            print("\nPercentage correctly classified characters using SMV: ")
            print("%.2f" % round(result*100,2))
        if DETECTION:
            window_size = 20
            pot_chars = detection(detection_filename, clf, window_size, HOG, SCALING)
            show_chars(pot_chars, window_size)
    if NN:
        clf = svm_fit(train_x, train_y)
        if CLASSIFICATION:
            result = svm_predict(clf, test_x, test_y)
            print("\nPercentage correctly classified characters using NN: ")
            print("%.2f" % round(result*100,2))
        if DETECTION:
            window_size = 20
            pot_chars = detection(detection_filename, clf, window_size, HOG, SCALING)
            show_chars(pot_chars, window_size)

#Inputs for running the program
SVM = True
NN = False
SCALING = True
HOG = True
CLASSIFICATION = False
DETECTION = True
detection_filename = 'detection-1.jpg'

run(SVM, NN, SCALING, HOG, CLASSIFICATION, DETECTION, detection_filename)
        



