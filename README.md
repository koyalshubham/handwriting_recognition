# packages to download:
openCV
os
math
numpy
scikit-image
tqdm

# input
As input this program has a folder named 'images' containing of the fused dead-sea scrolls

# goal
This program uses image processing to find the characters in the scroll. After that machine learning is used to name the characters.

# output image processing
As output this program has two folders.
segmented_characters contains the imgs with a character.
marked_areas contains the scrolls with the marked words and their line number. 

# output machine learning
The segemented characters for each folder of the scrolls were extracted into a training_data data structrue. Once this Data Structure was loaded. The training_data was later converted to a CSV which was the input for the recogniser created.
