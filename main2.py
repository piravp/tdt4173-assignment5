import os
from PIL import Image

# Declare constants
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
FOLDER = '/chars74k-lite'
PATH = ABS_PATH + FOLDER

# ...
files_path = [os.path.relpath(x) for x in os.listdir(PATH)]


for folder in files_path: 
    # List all folders
    print('Now considering folder', folder)
    print()

print(files_path)