import numpy as np
import cv2
import os
import math
from skimage import filters
from skimage import morphology
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv

#import matplotlib.pyplot as plt
# %matplotlib inline
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.optimizers import SGD
# from keras.layers.normalization import BatchNormalization
# from keras.utils import np_utils
# from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.preprocessing.image import ImageDataGenerator
# from keras.regularizers import l2


# used to sort contours from left to right top to bottom
def get_contour_rank(contour, cols):
    tolerance_factor = 20
    x, y, w, h = cv2.boundingRect(contour)
    y = y + h / 2  # take average height
    # // is floor division
    return ((y // tolerance_factor) * tolerance_factor) * cols + x


def convert_scroll_to_imgs():
    print("Converting scrolls to imgs")
    imagefolder_path = 'images'

    marked_words_path = 'marked_words'
    segmented_char_path = 'segmented_characters'

    directories = [marked_words_path, segmented_char_path]

    # make directories if they do not yet exist
    for path in directories:
        if not os.path.exists(path):
            os.mkdir(path)
            print("Directory " + str(path) + " created")

    images = os.listdir(imagefolder_path)
    for scroll in images:
        # skip hidden files (this is necessary for some filing systems)
        if not scroll.startswith('.'):
            # read image
            image_path = imagefolder_path + '/' + scroll
            im = cv2.imread(image_path)
            print("Finding characters in image: %s" % image_path)

            # set appropriate format for threshold function
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            ################### find scroll contour ######################
            # reduce noise
            imgray = cv2.medianBlur(imgray, 45)

            # get threshold value
            val = filters.threshold_otsu(imgray)

            # thresholding using value from otsu
            ret, binary = cv2.threshold(imgray, val, 255, cv2.THRESH_BINARY)
            (contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # sort contours descending
            contours.sort(key=len, reverse=True)

            # center of image
            imX = im.shape[1] / 2
            imY = im.shape[0] / 2

            # for the 4 largest contours, compute the distance to center of image
            distance = -1
            n_contour = min(4, len(contours))  # there are not always 4 contours
            for c in range(0, n_contour):
                # calculate moments of binary image
                M = cv2.moments(contours[c])

                # calculate x and y coordinate of center of mass
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                newDistance = math.sqrt(math.pow(imX - cX, 2) + math.pow(imY - cY, 2))

                if distance == -1 or newDistance < distance:
                    distance = newDistance
                    goodContour = c

            # keep contour
            mask = np.zeros(im.shape[:2], dtype="uint8")
            cv2.drawContours(mask, contours, goodContour, (255, 255, 255), cv2.FILLED)
            im = cv2.bitwise_or(im, im, mask=mask)

            bk = np.full(im.shape, 255, dtype=np.uint8)  # white bk
            mask = cv2.bitwise_not(mask)
            bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
            im = cv2.bitwise_or(im, bk_masked)

            ##################### remove holes #####################

            # set appropriate format for threshold function
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # find contour
            (_, thresh) = cv2.threshold(imgray, 15, 255, cv2.THRESH_BINARY)
            (contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # keep patch without holes
            mask = np.full(im.shape[:2], 255, dtype=np.uint8)
            for c in range(0, len(contours)):
                if (len(contours[c]) < 7000 and len(contours[c]) > 280):
                    # print("contour: %d", len(contours[c]))
                    cv2.drawContours(mask, contours, c, (0, 0, 0), cv2.FILLED)
            im = cv2.bitwise_or(im, im, mask=mask)

            bk = np.full(im.shape, 255, dtype=np.uint8)  # white bk
            mask = cv2.bitwise_not(mask)
            bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

            no_holes = cv2.bitwise_or(im, bk_masked)

            ##################### binarize image #####################
            image = cv2.imread(imagefolder_path + '/' + scroll)

            # set appropriate format for threshold function
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # get threshold value
            val = filters.threshold_otsu(image_gray)

            im_gray = cv2.cvtColor(no_holes, cv2.COLOR_BGR2GRAY)

            # binarize image without holes using value from otsu
            ret, binary = cv2.threshold(im_gray, val, 255, cv2.THRESH_BINARY_INV)

            # ################## remove small noise ########################

            # perform opening by reconstruction to remove small objects and maintain original characters
            # (erosion then dilation)
            kernel = np.ones((5, 5), np.uint8)
            img_erosion = cv2.erode(binary, kernel, iterations=1)
            marker = img_erosion.copy()
            mask = binary.copy()
            img_open_rec = morphology.reconstruction(marker, mask, method='dilation')

            ################## fill small holes ########################

            # perform closing on binary image to fill small holes (dilation then erosion)
            kernel = np.ones((5, 5), np.uint8)
            img_closed = cv2.dilate(img_open_rec, kernel, iterations=1)
            img_closed = cv2.erode(img_closed, kernel, iterations=1)
            img_closed_inv = cv2.bitwise_not(img_closed)

            ################## remove remaining noise by regular opening ###################

            # convert image to uint8 to pass to openCV functions
            img_closed = cv2.convertScaleAbs(img_closed)

            # remove more noise by erosion
            kernel = np.ones((3, 3), np.uint8)
            img_eroded2 = cv2.erode(img_closed, kernel, iterations=1)

            # ################## find words ########################

            # find contours
            (ctrs, hierarchy) = cv2.findContours(img_eroded2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # sort contours
            ctrs.sort(key=lambda x: get_contour_rank(x, img_eroded2.shape[1]))

            # get scroll name without extension
            folder = scroll[:-4]

            # make directory
            if not os.path.exists(segmented_char_path + '/' + folder):
                os.mkdir(segmented_char_path + '/' + folder)

            # copy read image
            image = np.copy(im)
            image_copy = np.copy(im)
            image_copy2 = np.copy(im)

            # get the actual inner list of hierarchy descriptions
            hierarchy = hierarchy[0]

            # initialize variables
            line = 0
            n_words = 0
            average_y = 0
            min_height = 20
            min_width = 20

            # loop over the found words
            for component in zip(ctrs, hierarchy):
                # Get bounding box
                currentContour = component[0]
                currentHierarchy = component[1]
                x, y, w, h = cv2.boundingRect(currentContour)
                if currentHierarchy[3] < 0:  # these are the outermost parent components

                    # Getting ROI
                    roi = image[y:y + h, x:x + w]
                    if h > min_height or w > min_width:  # if image is not noise
                        # determine line number line
                        if h < 100:  # normal segment
                            if y + h / 2 > average_y + 40 or y + h / 2 < average_y - 40:
                                # next line
                                line += 1
                                average_y = y + h / 2
                                n_words = 0
                            else:
                                # same line
                                average_y = (n_words * average_y + y + h / 2) / (n_words + 1)
                                n_words += 1

                        # draw pink bounding box with line-number
                        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (90, 0, 255), 2)
                        cv2.putText(image_copy, str(line), cv2.boundingRect(currentContour)[:2],
                                    cv2.FONT_HERSHEY_COMPLEX, 1, [125])

                        ############### Finding Characters ##############
                        shifted = cv2.pyrMeanShiftFiltering(roi.copy(), 90, 130)
                        # set appropriate format for threshold function
                        imgray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

                        # reduce noise
                        factor = 5
                        kernel = np.ones((factor, factor), np.float32) / (factor * factor)
                        imgray = cv2.filter2D(imgray, -1, kernel)

                        # binarize
                        (_, thresh) = cv2.threshold(imgray, 125, 255, cv2.THRESH_BINARY)

                        # make white border
                        thresh = cv2.copyMakeBorder(thresh, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

                        # find contour
                        (contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                        # sort contours on size area
                        sortedContours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

                        char_number = 'a'
                        for c in range(1, len(sortedContours)):
                            if cv2.contourArea(sortedContours[c]) > 30:
                                imcopy = roi.copy()

                                # store character
                                mask = np.zeros(imcopy.shape[:2], dtype="uint8")
                                cv2.drawContours(mask, sortedContours, c, (255, 255, 255), cv2.FILLED)
                                imcopy = cv2.bitwise_or(imcopy, imcopy, mask=mask)

                                bk = np.full(imcopy.shape, 255, dtype=np.uint8)  # white bk
                                mask = cv2.bitwise_not(mask)
                                bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
                                imcopy = cv2.bitwise_or(imcopy, bk_masked)

                                # filter out noise based on mean and stdev
                                x2, y2, w2, h2 = cv2.boundingRect(sortedContours[c])
                                character = roi[y2:y2 + h2, x2:x2 + w2]

                                mean = np.mean(character)
                                stdev = np.std(character)
                                mean_min = 50
                                mean_max = 189
                                stdev_min = 22
                                stdev_max = 115

                                # remove noise images
                                if mean >= mean_min and mean <= mean_max and stdev >= stdev_min and stdev <= stdev_max:
                                    # save characters
                                    x_char = x + x2
                                    y_char = y + y2

                                    cv2.imwrite(
                                        segmented_char_path + '/' + folder + '/line ' + str(line) + ' col ' + str(
                                            x_char) + '.jpg', character)

                                    # draw rectangle around characters
                                    cv2.rectangle(image_copy2, (x_char, y_char), (x_char + w2, y_char + h2),
                                                  (0, 255, 0), 2)

            # save marked words
            cv2.imwrite(marked_words_path + '/' + scroll, image_copy)

    print("Completed scroll to img")


def convert_img_to_csv():

    ###---- START -> to extract the line and col nos from the filenames
    parent_folder_name = "segmented_characters"
    segmented_folder_info = []

    for subdir, dirs, files in os.walk(parent_folder_name):
    
        for folder in dirs:
            filepath = "{}/{}".format(parent_folder_name, folder)
            
            line_list = []
            line_no_list = []
            col_list = []
            col_no_list = []
            filename_list = []
            for file in os.listdir(filepath):
                image_filepath = "{}/{}".format(filepath, file)
                data = file.split(" ")
                line_list.append(data[0])
                line_no_list.append(int(data[1]))
                col_list.append(data[2])
                col_no_list.append(int(data[3][:-4]))
                filename_list.append(image_filepath)
                
            data = {
                'line':line_list,
                'line_no':line_no_list,
                'col':col_list,
                'col_no':col_no_list,
                'filename':filename_list
            }
            df = pd.DataFrame(data)
            df = df.sort_values(['line_no', 'col_no'], ascending=[True, True])
            segmented_folder_info.append([filepath, df])

    ###---- END -> to extract the line and col nos from the filenames

    for scroll_folder_name, df in segmented_folder_info:

        scroll_array = []

        for index, row in df.iterrows():
            training_data = []
            img_filename = row['filename']

            try:
                IMG_SIZE = 100
                img_array = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

                training_data.append([new_array, img_filename])
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, img_filename)
            except Exception as e:
                print("general exception", e, img_filename)
            
            X = []
            y = []

            for features, label in training_data:
                X.append(features)
                y.append(label)

            X_test = []
            for img in X:
                new_img = np.reshape(img, 10000)
                X_test.append(new_img)

            X_test = np.asarray(X_test)
            scroll_array.append(X_test)

        print(len(scroll_array))

        with open("{}.csv".format(scroll_folder_name),'w') as f1:
            writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
            
            for i in scroll_array:            
                writer.writerow(i[0])

    print("Completed img to csv")

def make_training_csv():
    DATADIR = "training-data"

    CATEGORIES = ["Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final", "Lamed", "Mem",
                  "Mem-medial", "Nun-final", "Nun-medial", "Pe", "Pe-final", "Qof", "Resh", "Samekh", "Shin", "Taw",
                  "Tet", "Tsadi-final", "Tsadi-medial", "Waw", "Yod", "Zayin"]

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        print(path)
        os.listdir(path)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            # plt.imshow(img_array, cmap='gray')
            # plt.show()  # display!

            # break
        # break

    IMG_SIZE = 100

    training_data = create_training_data()

    sample_labels = np.zeros(len(training_data))
    for index, sample in enumerate(training_data):
        sample_labels[index] = sample[1]

    np.arange(27)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    # to dump all data into a csv for X_train

    X_train = []
    for img in X:
        new_img = np.reshape(img, 10000)
        new_img = new_img.tolist()
        X_train.append(new_img)

    with open('X.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(X_train)

    with open('y.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        for label in y:
            writer.writerow([label])


def create_training_data():
    training_data = []

    DATADIR = "training-data"

    CATEGORIES = ["Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final", "Lamed", "Mem",
                  "Mem-medial", "Nun-final", "Nun-medial", "Pe", "Pe-final", "Qof", "Resh", "Samekh", "Shin", "Taw",
                  "Tet", "Tsadi-final", "Tsadi-medial", "Waw", "Yod", "Zayin"]

    IMG_SIZE = 100

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # get the classification  (0 to 26).

        for img in tqdm(os.listdir(path)):  # iterate over each image per category

            try:

                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data

            except Exception as e:  # in the interest in keeping the output clean...
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path, img))
            except Exception as e:
                print("general exception", e, os.path.join(path, img))
    return training_data


# def run_model():
#     with open('X.csv', 'r') as infile:
#         reader = csv.reader(infile)
#         lines = list(reader)

#     X = []
#     for row in lines:
#         new_row = [float(i) for i in row]
#         X.append(new_row)

#     X = np.asarray(X)

#     with open('y.csv', 'r') as infile:
#         reader = csv.reader(infile)
#         lines = list(reader)

#     y = []
#     for row in lines:
#         new_row = [float(i) for i in row]
#         y.append(new_row)

#     y = np.asarray(y)

#     X_train = X.reshape(X.shape[0], 100, 100, 1)
#     X_train = X_train.astype('float32')
#     X_train /= 255

#     number_of_classes = 27

#     Y_train = np_utils.to_categorical(y, number_of_classes)

#     # Miniaturized VGG-16 model

#     model = Sequential()

#     model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 1), activation='relu', padding='same'))
#     # model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     # model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(256, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#     model.add(Flatten())

#     model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
#     model.add(Dropout(0.5))
#     model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
#     model.add(Dropout(0.5))
#     model.add(Dense(27, activation='softmax'))

#     model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

#     model.summary()

#     model.fit(X_train, Y_train, batch_size=128, epochs=50, verbose=1)

#     with open('test.csv', 'r') as infile:
#         reader = csv.reader(infile)
#         lines = list(reader)

#     X_test = []
#     for row in lines:
#         new_row = [float(i) for i in row]
#         X_test.append(new_row)

#     X_test = np.asarray(X_test)

#     predictions = model.predict(X_test)

#     np.savetxt('predictions.txt', predictions, delimiter=',')


# convert scroll img to character imgs
convert_scroll_to_imgs()
# convert imgs to csv
convert_img_to_csv()
make_training_csv()
# run model
#run_model()
