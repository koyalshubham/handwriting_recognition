# packages to download:

Please run the requirements.sh file with sudo command to install all the required packages.

# packages used

openCV
os
math
numpy
scikit-image
tqdm
keras
sys
h5py

# input
As input this program takes the path containing the fused dead-sea scrolls given by the user as an argument using command. Example: python my_classifier.py mypath/testset/.
If no path is given, the path takes the default value 'images' and searches for a folder with that name in the same root directory.


# goal
This program uses image processing to find the characters in the scroll. After that machine learning is used to name the characters.

# output image processing
As output this program has two folders.
segmented_characters contains the imgs with a character.
marked_areas contains the scrolls with the marked words and their line number. 

# output machine learning
The segemented characters for each folder of the scrolls were extracted into a training_data data structrue. Once this Data Structure was loaded. The training_data was later converted to a CSV which was the input for the recogniser created.
