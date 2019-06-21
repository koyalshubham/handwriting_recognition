import cv2
import numpy as np
import os
from scipy import ndimage
from skimage import img_as_ubyte


def character_segmentation(foldername):

    imagefolder_path = 'segmented_areas/'+foldername
    savefolder_path = 'shubhamtest'
    blur_radius = 1
    threshold = 100

    # make directories if needed
    directories = [savefolder_path]
    for path in directories:
        if not os.path.exists(path):
            os.mkdir(path)
            print("Directory ", path, " created ")

    # for all images in imagefolder
    images = os.listdir(imagefolder_path)
    for scroll in images:
	    # read image
	    image_path = imagefolder_path + '/' + scroll
	    im = cv2.imread(image_path)
	    print("Processing image: %s" %image_path)
	    shifted = cv2.pyrMeanShiftFiltering(im, 90, 130)
	    # set appropriate format for treshold function
	    imgray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
	
	    # reduce noise
	    factor = 5
	    kernel = np.ones((factor,factor), np.float32)/(factor*factor)
	    imgray = cv2.filter2D(imgray, -1, kernel)
	
	    # binarize
	    ( _, thresh) = cv2.threshold(imgray, 125, 255, cv2.THRESH_BINARY)

	    #connected components
	    imgf = ndimage.gaussian_filter(thresh, blur_radius)
	    labeled, nr_objects = ndimage.label(imgf > threshold)
	    #labeled = cv2.cvtColor(labeled, cv2.COLOR_BGR2GRAY)
	    cv2.normalize(labeled, labeled, 0, 255, cv2.NORM_MINMAX)

	    # make white border
	    thresh = cv2.copyMakeBorder(labeled,1,1,1,1,cv2.BORDER_CONSTANT,value=255)  #change label to thresh here
	    """ print(thresh.dtype) """
	    thresh = img_as_ubyte(thresh)

	    # find contour
	    print(thresh.dtype)
	    (contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	    # for a newer version of openCV, use the following:
	    # (_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
	    # sort contours on size area
	    sortedContours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)
	
	    char_number = 'a'
	    for c in range(1, len(sortedContours)):
		    if cv2.contourArea(sortedContours[c]) > 35:
			    imcopy = im.copy()
			
			    # store character
			    mask = np.zeros(imcopy.shape[:2], dtype="uint8")
			    cv2.drawContours(mask, sortedContours, c, (255,255,255), cv2.FILLED)
			    imcopy = cv2.bitwise_or(imcopy, imcopy, mask=mask)

			    bk = np.full(imcopy.shape, 255, dtype=np.uint8)  # white bk
			    mask = cv2.bitwise_not(mask)
			    bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
			    imcopy = cv2.bitwise_or(imcopy, bk_masked)
			
			    cv2.imwrite(savefolder_path + '/' + scroll + '_' + char_number + '.png', imcopy)
			    char_number = chr(ord(char_number) + 1)

    return 1