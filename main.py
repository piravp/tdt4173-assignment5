import os
import numpy as np
from PIL import Image
from sklearn import preprocessing as pp
import random
from sklearn.svm import SVC
from collections import OrderedDict
import skimage.feature as skimage
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Loads images from the chars74k-lite dataset and loads it into a dictionary
# flatten - boolean variable, set to True if the images needs to be flattened
# returns images, a dictionary with a np-array with np-arrays
def load_images(flatten):
    images = OrderedDict()
    first = True
    # Iterate over each character-folder
    for folder in os.walk(FULL_PATH):
        # First iteration is over the root folder 
        if first:
            first = False
            continue
        # Path to the characterfolder
        letter_path = folder[0]
        letter = letter_path[-1]
        idx = 0
        letters = [[]]
        # Iterate over each image
        for file in os.listdir(letter_path):
            if file.endswith(".jpg"):
                # Open the image and convert it into a numpy array
                image = Image.open(letter_path + '/' + file)
                pix = np.array(image)
                if flatten:
                    pix = pix.flatten()
                # Scale pixels to 0:1 range
                letters.append(pix/255)
                idx += 1
        # Remove the first instance, which is empty
        images[letter] = np.array(letters)[1:]
    return images
            
# Preprocessing using scaling to have unit variance and 0 mean
def pre_scaling(images):
    for letter in images:
        for img_idx in range (0, images[letter].size):
            img = images[letter][img_idx]
            images[letter][img_idx] = pp.scale(img)
    return images

# Preprocessing using Histogram of Oriented Gradients
def pre_hog(images):
    for letter in images:
        for idx in range(0,len(images[letter])):
            img = images[letter][idx]
            images[letter][idx] = skimage.hog(img, orientations = 10,pixels_per_cell=(5,5), cells_per_block=(1,1))
    return images

# Divide the dataset into training data and test data
# Modifies images into training data
# Returns test data
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

# Fit SVM (SVC) classifier
def svm_fit(x,y):
    clf = SVC(probability=True)
    clf.fit(x,y)
    return clf

# Predict values for x using a fitted classifier and comparing the results to
# the label. Returning percentage correclty classified images
def predict(clf, test_x , test_y):
    correct = 0
    total = 0
    for i in range(0, len(test_x)):
        y_predicted = clf.predict([test_x[i]])
        if y_predicted == test_y[i]:
            correct += 1
        total += 1
    return correct/total

# Create feature-label pairs to be passed to classifiers    
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
    
# Fit a Neural Network classifier
def nn_fit(x,y):
    clf = MLPClassifier(solver='lbfgs', alpha=0.05, hidden_layer_sizes=(80, 40), random_state=1, learning_rate_init = 0.2, learning_rate = 'adaptive')
    clf.fit(x,y)
    return clf

# Load image used for detection
def load_detection_image(filename):
    image = Image.open(filename)
    pix = np.array(image)
    return pix

# Detect characters in an image, using sliding window    
def detection(filename, clf, window_size, HOG, SCALING):
    print("Detecting...")
    threshold = 0.6                            # Threshold for including image
    image = load_detection_image(filename)
    max_row = image.shape[0]
    max_col = image.shape[1]
    pot_chars = []
    # Iterate over the rows and columns of the image
    for row in range(0,max_row-window_size,1):
        for col in range(0,max_col-window_size,1):
            # Define and scale partial image
            sub_img = image[np.ix_(range(row,row+window_size),range(col,col+window_size))]
            if check_edges(sub_img):
                old_sub_img = sub_img/255
                for i in range (0,4):
                    sub_img = np.rot90(old_sub_img, k=i)
                    if HOG:
                        sub_img = skimage.hog(sub_img, orientations = 10,pixels_per_cell=(5,5), cells_per_block=(1,1))
                    if SCALING:
                        sub_img = pp.scale(sub_img)
                    # HOG flattens, so if not used it has to be done manually
                    if not HOG:
                        sub_img = sub_img.flatten()
                    # Predict probabilities
                    res = clf.predict_proba([sub_img])
                    res_max = 0
                    idx = 0
                    # Find label with highest probability
                    for i in range(0,26):
                        if res[0][i] > res_max:
                            res_max = res[0][i]
                            idx = i
                    # If the probability is above the threshold, add to list of 
                    # potential characters in the image
                    if res_max > threshold:
                        pot_chars.append((row,col,idx,res_max))
                        break
    # Refine potential characters to limit neighbours
    # Inside if-sentence for debugging purposes
    if True:
        pot_chars = refine_chars(pot_chars)
    return pot_chars, image

def check_edges(img):
    x = img.shape[0]
    y = img.shape[1]
    top = 0
    bottom = 0
    left = 0
    right = 0
    for i in range (0,x):
        if img[i][2] == 255:
            top +=1
        if img[i][y-3] == 255:
            bottom +=1
    for i in range(0,y):
        if img[2][i] == 255:
            left+=1
        if img[x-3][i] == 255:
            right +=1
    return (top < 3 and bottom < 3 and left < 5 and right <5)
            

# Removes potential characters close to eachother
def refine_chars(pot_chars):
    new_chars = pot_chars[:]
    to_delete = []
    # Double loop through all potential characters
    for idx, fig in enumerate(pot_chars):
        row = fig[0]
        col = fig[1]
        for new_idx, new_fig in enumerate(new_chars):
            new_row = new_fig[0]
            new_col = new_fig[1]
            # If they are sufficiently close and one has suppirior value, delete the other
            if abs(row-new_row) < 11 and abs(col-new_col) < 11 and idx != new_idx:
                if fig[3] > new_fig[3]:
                    to_delete.append(new_idx)
    # Delete all characters identified to be shaved off
    to_delete = list(set(to_delete))
    li = sorted(to_delete, reverse=True)
    for i in li:
        del pot_chars[i]                      
    return pot_chars

# Plot detection image and boxes around identified characters
def show_chars(image, pot_chars, window_size):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    for fig in pot_chars:
        row = fig[0]
        col = fig[1]
        rect = patches.Rectangle((col-1,row-1),20,20,linewidth=1,edgecolor='r',facecolor='none')
        plt.text(col+20, row, chr(fig[2]+97), fontsize=12)
        ax.add_patch(rect)
    plt.show()
   
# Predict the label probabilities for a single image
def single_prediction(clf, image,label):
    res = clf.predict_proba([image])
    idx = 0
    res_max = 0
    # Find highest scoring label
    for i in range(0,26):
        if res[0][i] > res_max:
            res_max = res[0][i]
            idx = i
    print('\nPredicted letter ' +str(chr(idx+97))+' with probability '+str(res_max))
    print('Correct label: '+str(label))
    
# Main method used for running the code
# Each input variable defines which methods should be used
# Last input is filename for detection image
def run(SVM, NN, SCALING, HOG, CLASSIFICATION, DETECTION, detection_filename):
    clf = None
    print("Loading...")
    # Load images
    images = load_images(not HOG)
    # Preprocess images
    if HOG:
        images = pre_hog(images)
    if SCALING:
        images = pre_scaling(images)
    # Divide dataset
    test_images = divide_dataset(images, 0)
    # Make the dataset into a suitable format for the classifiers
    train_x, train_y = create_labels(images)
    test_x, test_y = create_labels(test_images)
    print("Fitting...")
    # If SVM, fit SVM (SVC) classifier, predict and print results
    if SVM:
        clf = svm_fit(train_x, train_y)
        if CLASSIFICATION:
            result = predict(clf, train_x, train_y)
            print("\nPercentage correctly classified characters from training set using SVM: ")
            print("%.2f" % round(result*100,2))
            result = predict(clf, test_x, test_y)
            print("\nPercentage correctly classified characters from test set using SVM: ")
            print("%.2f" % round(result*100,2))
        # If DETECTION, use classifier to detect characters in an image
        if DETECTION:
            window_size = 20
            pot_chars, image = detection(detection_filename, clf, window_size, HOG, SCALING)
            show_chars(image, pot_chars, window_size)
    # If NN, fit NN classifier, predict and print results
    if NN:
        clf = nn_fit(train_x, train_y)
        if CLASSIFICATION:
            result = predict(clf, train_x, train_y)
            print("\nPercentage correctly classified characters from training set using NN: ")
            print("%.2f" % round(result*100,2))
            result = predict(clf, test_x, test_y)
            print("\nPercentage correctly classified characters from test set using NN: ")
            print("%.2f" % round(result*100,2))
        # If DETECTION, use classifier to detect characters in an image
        if DETECTION:
            window_size = 20
            pot_chars, image = detection(detection_filename, clf, window_size, HOG, SCALING)
            show_chars(image, pot_chars, window_size)
    return clf

# Path for chars74k-lite folder
# Detection images in same folder as the program
PATH = os.path.dirname(os.path.abspath(__file__))
FOLDER = '/chars74k-lite'
FULL_PATH = PATH + FOLDER

# Inputs for running the program
SVM = True             # Fit classifier using SVM (SVC)
NN = False               # Fit classifier using Neural Networks
SCALING = True          # Use scaling for preprocessing     
HOG = True              # Use Histogram of Oriented Gradients for preprocessing
CLASSIFICATION = False   # Use classifier to predict labels of images   
DETECTION = True       # Use classifier to detect multiple characters in an image
SINGLES = False          # Single image probability predictions
detection_filename = 'detection-2.jpg'  # Filename for the file with characters to be detected
# Having both NN and DETECTION set to True requires substantil processing time

# MAIN FUNCTION
clf = run(SVM, NN, SCALING, HOG, CLASSIFICATION, DETECTION, detection_filename)
      
# SINGLE PREDICTION
if SINGLES:
    images = load_images(not HOG)
    if HOG:
        images = pre_hog(images)
    if SCALING:
        images = pre_scaling(images)
        num = 10
        single_prediction(clf,images['a'][num],'a')
        single_prediction(clf,images['e'][num],'e')  
        single_prediction(clf,images['g'][num],'g')  
        single_prediction(clf,images['k'][num],'k')  
        single_prediction(clf,images['r'][num],'r')  
        single_prediction(clf,images['t'][num],'t')
        single_prediction(clf,images['w'][num],'w')    



