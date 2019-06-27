# packages to download

Please run the requirements.sh file with sudo command to install all the required packages.
This can be done with the following command:<br/>
chmod +x requirements.sh<br/>
sudo ./requirements.sh<br/>
<br/>
You may have to manually install opencv-python, pandas, keras, tensorflow(1.14), tpdm, scikit-image, especially if you run it on a virtualenv. Please use tensorflow 1.X versions only.

# input
As input this program takes the path containing the fused dead-sea scrolls given by the user as an argument using command. Example: python my_classifier.py mypath/testset/.
If no path is given, the path takes the default value 'images' and searches for a folder with that name in the same root directory.

# goal
This program uses image processing to find the characters in the scroll. After that machine learning is used to name the characters.

# function take_input_argument()
take_input_argument() sets the directory of the test images given by the user using command. If no path is given by the user, the path takes the default value 'images' and searches for a folder with that name in the same root directory.

# function convert_scroll_to_imgs(imagefolder_path)
convert_scroll_to_imgs(imagefolder_path) converts the test images in directory imagefolder_path to images of segmented characters. Each with the line number and column number as label. These images are stored in the root directory in folder 'segmented_characters' in a subfolder with the name of the test image. If theses folders do not exist then they will be created.

# function convert_img_to_csv():
convert_img_to_csv() converts the imgs of the segmented characters into a csv in the MNIST-like format, with a resolution of 50x50 pixels.

# function run_model()
run_model() loads the trained VGGNet and makes predictions on the csv generated from the segmented characters.

# output image processing
As output this program has two folders.
segmented_characters contains the imgs with a character.
marked_areas contains the scrolls with the marked words and their line number. 

# output machine learning
The segemented characters for each folder of the scrolls were extracted into a input_data data structure, which contains both the character id and the vector representation of the character. After preprocessing the training data into the proper format, it was fed to the model. As output for every scroll img, we get a txt with the top-5 predictions for every line and every character. The txt files are created in the parent directory.
