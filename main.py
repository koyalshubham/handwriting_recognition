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
import sys
from keras.models import load_model
from keras.models import Model


# used to sort contours from left to right top to bottom
def get_contour_rank(contour, cols):
    tolerance_factor = 20
    x, y, w, h = cv2.boundingRect(contour)
    y = y + h / 2  # take average height
    # // is floor division
    return ((y // tolerance_factor) * tolerance_factor) * cols + x

# used to segment characters from the test images
def convert_scroll_to_imgs(imagefolder_path):
    print("Converting scrolls to imgs")

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

# used to convert segmented characters into csv with pixel values per line and column
def convert_img_to_csv():
    ###---- START -> to extract the line and col nos from the filenames
    parent_folder_name = "segmented_characters"
    segmented_folder_info = []

    for subdir, dirs, files in os.walk(parent_folder_name):
        # convert information
        for folder in dirs:
            filepath = "{}/{}".format(parent_folder_name, folder)

            line_no_list = []
            col_no_list = []
            filename_list = []
            for file in os.listdir(filepath):
                image_filepath = "{}/{}".format(filepath, file)
                data = file.split(" ")
                line_no_list.append(int(data[1]))
                col_no_list.append(int(data[3][:-4]))
                filename_list.append(image_filepath)

            data = {
                'line_no': line_no_list,
                'col_no': col_no_list,
                'filename': filename_list
            }
            df = pd.DataFrame(data)
            df = df.sort_values(['line_no', 'col_no'], ascending=[True, True])
            segmented_folder_info.append([filepath, df])

    ###---- END -> to extract the line and col nos from the filenames

    for scroll_folder_name, df in segmented_folder_info:

        scroll_array = []
        for index, row in df.iterrows():
            test_data = []
            img_filename = row['filename']

            try:
                IMG_SIZE = 100
                img_array = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

                test_data.append(new_array)
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            except OSError as e:
                print("OSErrroBad img most likely", e, img_filename)
            except Exception as e:
                print("general exception", e, img_filename)

            line = row['line_no']
            col = row['col_no']

            char_array = []
            char_array.extend([line, col])

            for features in test_data:
                new_img = np.reshape(features, 10000)
                char_array.extend(new_img)

            scroll_array.append(char_array)

        with open("{}.csv".format(scroll_folder_name), 'w') as f1:
            writer = csv.writer(f1, lineterminator='\n', )

            for i in scroll_array:
                writer.writerow(i)

    print("Completed img to csv")

# used to predict the label of the segmented characters and generate a csv file with the prediction
def run_model():
    seg_chars = os.listdir('segmented_characters')
    for filename in seg_chars:
        if filename[-3:] == 'csv':
            model = load_model('hwr50px.h5')

            # preprocess input into train and char_id nparrays
            with open(filename, 'r') as infile:
                reader = csv.reader(infile)
                lines = list(reader)
            input_data = []
            for row in lines:
                new_row = [float(i) for i in row]
                input_data.append(new_row)
            input_data = np.asarray(input_data)

            char_id = input_data[:, :2]
            train = input_data[:, 2:]

            predictions = model.predict(train)

            # generate top5 predictions
            values = ["alef", "ayin", "bet", "dalet", "gimel", "he", "het", "kaf", "kaf-final", "lamed", "mem",
                      "mem-medial", "nun-final", "nun-medial", "pe", "pe-final", "qof", "resh", "samekh", "shin", "taw",
                      "tet", "tsadi-final", "tsadi-medial", "waw", "yod", "zayin"]
            keys = range(len(values))
            alphabets = dict(zip(keys, values))
            sorting = (-predictions).argsort()
            top5 = sorting[:, :5]

            # top5_predictions is a list of lists containing the top5 predictions per character
            # todo append char_id to each list of top5 predictions
            top5_predictions = []
            for i in range(top5.shape[0]):
                new_char = []
                for j in range(top5.shape[1]):
                    label_class = top5[i, j]
                    alphabet = alphabets[label_class]
                    new_char.append(alphabet)
                top5_predictions.append(new_char)

            top5_predictions = np.asarray(top5_predictions)
            predictions_with_id = np.concatenate((char_id, top5_predictions), axis=1)

            outfile_name = filename + ".txt"
            outfile = open(outfile_name, "w+")
            for prediction in range(1,6): # top 5 predictions
                outfile.write("Prediction " + str(prediction) + ":")
                for line in range(predictions_with_id.shape[0]):
                    if line == 1:
                        outfile.write(str(predictions_with_id[line,prediction+1]) + " ")
                    else:
                        if predictions_with_id[line-1,0] == predictions_with_id[line,0]:
                            outfile.write(str(predictions_with_id[line,prediction+1]) + " ")
                        else:
                            outfile.write("\r\n")
                            outfile.write(str(predictions_with_id[line,prediction+1]) + " ")
                outfile.write("\r\n")


# used to set the path to the test images
def take_input_argument():
    # print the given arguments
    print("The arguments are: ", str(sys.argv))
    if len(sys.argv) == 1:
        imagefolder_path = 'images'
        print("use default folder: " + imagefolder_path)
    else:
        imagefolder_path = sys.argv[1]
        print("given folder: " + imagefolder_path)
    return imagefolder_path

# take input argument (file path) from command
imagefolder_path = take_input_argument()
# convert scroll img to character imgs
convert_scroll_to_imgs(imagefolder_path)

# convert imgs to csv
convert_img_to_csv()

# run model
run_model()
