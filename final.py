import numpy as np
import cv2
import os
import math
from skimage import filters
from skimage import morphology

imagefolder_path = 'images'
scroll_path = 'scrolls_only'
holes_path = 'no_holes'
binary_path = 'binarized'
marked_areas_path = 'marked_areas'
segmented_areas_path = 'segmented_areas'
segmented_char_path = 'segmented_characters'
marked_char_path = 'marked_characters'

binary_otsu_path = 'binary_OTSU'
erosion_path = 'erosion'
closing_path = 'closing'
open_rec_path = 'opening_by_reconstruction'
erosion_after_rec_path = 'erosion_after_reconstruction'
dilation_path = 'dilated_areas'

segmented_areas_path_trash = 'segmented_areas_trash'  # used for debugging, remove in the end

directories = [scroll_path, holes_path, binary_path, marked_areas_path, segmented_areas_path,
               binary_otsu_path, erosion_path, closing_path, open_rec_path, erosion_after_rec_path,
               dilation_path,
               segmented_areas_path_trash, segmented_char_path, marked_char_path]

# make directories if they do not yet exist
for path in directories:
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory " + str(path) + " created")


# used to sort contours from left to right top to bottom
def get_contour_rank(contour, cols):
    tolerance_factor = 20
    x, y, w, h = cv2.boundingRect(contour)
    y = y + h / 2  # take average height
    # // is floor division
    return ((y // tolerance_factor) * tolerance_factor) * cols + x


images = os.listdir(imagefolder_path)
for scroll in images:
    # skip hidden files (this is necessary for some filing systems)
    if not scroll.startswith('.'):
        # read image
        image_path = imagefolder_path + '/' + scroll
        im = cv2.imread(image_path)
        print("Processing image: %s" % image_path)

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

            # calculate x,y coordinate of center of mass
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

        # save image
        cv2.imwrite(scroll_path + '/' + scroll, im)

        ##################### remove holes #####################
        im = cv2.imread(scroll_path + '/' + scroll)

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
        im = cv2.bitwise_or(im, bk_masked)

        # save image
        cv2.imwrite(holes_path + '/' + scroll, im)

        ##################### binarize image #####################
        image = cv2.imread(imagefolder_path + '/' + scroll)

        # set appropriate format for threshold function
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # get threshold value
        val = filters.threshold_otsu(image_gray)

        im = cv2.imread(holes_path + '/' + scroll)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # binarize image without holes using value from otsu
        ret, binary = cv2.threshold(im_gray, val, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(binary_otsu_path + '/' + scroll, binary)

        # ################## remove small noise ########################

        # perform opening by reconstruction to remove small objects and maintain original characters
        # (erosion then dilation)
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(binary, kernel, iterations=1)
        cv2.imwrite(erosion_path + '/' + scroll, img_erosion)
        marker = img_erosion.copy()
        mask = binary.copy()
        img_open_rec = morphology.reconstruction(marker, mask, method='dilation')
        cv2.imwrite(open_rec_path + '/' + scroll, img_open_rec)

        ################## fill small holes ########################

        # perform closing on binary image to fill small holes (dilation then erosion)
        kernel = np.ones((5, 5), np.uint8)
        img_closed = cv2.dilate(img_open_rec, kernel, iterations=1)
        img_closed = cv2.erode(img_closed, kernel, iterations=1)
        img_closed_inv = cv2.bitwise_not(img_closed)
        # save image
        cv2.imwrite(closing_path + '/' + scroll, img_closed)

        ################## remove remaining noise by regular opening ###################

        # convert image to uint8 to pass to openCV functions
        img_closed = cv2.convertScaleAbs(img_closed)

        # remove more noise by erosion
        kernel = np.ones((3, 3), np.uint8)
        img_eroded2 = cv2.erode(img_closed, kernel, iterations=1)
        cv2.imwrite(erosion_after_rec_path + '/' + scroll, img_eroded2)

        # ################## find areas ########################

        # find contours
        (ctrs, hierarchy) = cv2.findContours(img_eroded2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        ctrs.sort(key=lambda x: get_contour_rank(x, img_eroded2.shape[1]))

        # get scroll name without extension
        folder = scroll[:-4]

        if not os.path.exists(segmented_areas_path + '/' + folder):
            os.mkdir(segmented_areas_path + '/' + folder)
            print("Directory ", segmented_areas_path + '/' + folder, " created ")

        if not os.path.exists(segmented_areas_path_trash + '/' + folder):
            os.mkdir(segmented_areas_path_trash + '/' + folder)
            print("Directory ", segmented_areas_path_trash + '/' + folder, " created ")

        if not os.path.exists(segmented_char_path + '/' + folder):
            os.mkdir(segmented_char_path + '/' + folder)
            print("Directory ", segmented_char_path + '/' + folder, " created ")

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

        # loop over the found areas
        for component in zip(ctrs, hierarchy):
            # Get bounding box
            currentContour = component[0]
            currentHierarchy = component[1]
            x, y, w, h = cv2.boundingRect(currentContour)
            if currentHierarchy[3] < 0:  # these are the outermost parent components

                # Getting ROI
                roi = image[y:y + h, x:x + w]

                if h <= min_height and w <= min_width:  # if image is noise
                    cv2.imwrite(
                        segmented_areas_path_trash + '/' + folder + '/line ' + str(line) + ' col ' + str(x) + '.jpg',
                        roi)

                else:  # if image is not noise

                    # determine line number line
                    if h < 100:
                        # normal segment
                        if y + h / 2 > average_y + 40 or y + h / 2 < average_y - 40:
                            # next line
                            line += 1
                            average_y = y + h / 2
                            n_words = 0
                        else:
                            # same line
                            average_y = (n_words * average_y + y + h / 2) / (n_words + 1)
                            n_words += 1

                    # reset values
                    old_y = y
                    old_h = h
                    old_x = x

                    # Save segments
                    cv2.imwrite(segmented_areas_path + '/' + folder + '/line ' + str(line) + ' col ' + str(x) + '.jpg',
                                roi)

                    # draw a rectangle around the segmented area
                    if (h > 100):
                        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (90, 0, 255), 2)

                    # draw number of contour
                    cv2.putText(image_copy, str(line), cv2.boundingRect(currentContour)[:2], cv2.FONT_HERSHEY_COMPLEX,
                                1, [125])

                    ###############Finding Characters##############
                    #print("Finding characers")
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
                    # for a newer version of openCV, use the following:
                    # (_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    # sort contours on size area
                    sortedContours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

                    char_number = 'a'
                    for c in range(1, len(sortedContours)):
                        if cv2.contourArea(sortedContours[c]) > 30:
                            imcopy = roi.copy()

                            #print("In if")
                            # store character
                            mask = np.zeros(imcopy.shape[:2], dtype="uint8")
                            cv2.drawContours(mask, sortedContours, c, (255, 255, 255), cv2.FILLED)
                            imcopy = cv2.bitwise_or(imcopy, imcopy, mask=mask)

                            bk = np.full(imcopy.shape, 255, dtype=np.uint8)  # white bk
                            mask = cv2.bitwise_not(mask)
                            bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
                            imcopy = cv2.bitwise_or(imcopy, bk_masked)

                            #save characters
                            x2, y2, w2, h2 = cv2.boundingRect(sortedContours[c])
                            x_char = x + x2
                            y_char = y + y2
                            cv2.imwrite(segmented_char_path + '/' + folder + '/line ' + str(line) + ' col ' + str(x_char) + '.jpg',
                            imcopy)

                            #draw rectangle around characters
                            cv2.rectangle(image_copy2, (x_char, y_char), (x_char + w2, y_char + h2), (0, 255, 0), 2)


        # save marked areas
        cv2.imwrite(marked_areas_path + '/' + scroll, image_copy)

        # save marked characters
        cv2.imwrite(marked_char_path + '/' + scroll, image_copy2)
