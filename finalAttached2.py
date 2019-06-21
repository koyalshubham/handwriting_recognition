import numpy as np
import cv2
import os
import math
from skimage import filters
from skimage import morphology
from tqdm import tqdm
        
#used to sort contours from left to right top to bottom
def get_contour_rank(contour, cols):
	tolerance_factor = 20
	x, y, w, h = cv2.boundingRect(contour)
	y = y + h / 2	# take average height
	# // is floor division
	return ((y // tolerance_factor) * tolerance_factor) * cols + x

def convert_scroll_to_imgs():
	print("Converting scrolls to imgs")
	imagefolder_path = 'images'
	scroll_path = 'scrolls_only'
	holes_path = 'no_holes'
	binary_path = 'binarized'
	marked_areas_path = 'marked_areas'
	segmented_areas_path = 'segmented_areas'

	binary_otsu_path = 'binary_OTSU'
	erosion_path = 'erosion'
	closing_path = 'closing'
	open_rec_path = 'opening_by_reconstruction'
	erosion_after_rec_path = 'erosion_after_reconstruction'
	dilation_path = 'dilated_areas'

	segmented_areas_path_trash = 'segmented_areas_trash'  # used for debugging, remove in the end


	directories = [	scroll_path, holes_path, binary_path, marked_areas_path, segmented_areas_path,
		binary_otsu_path, erosion_path, closing_path, open_rec_path, erosion_after_rec_path,
		dilation_path,
		segmented_areas_path_trash]

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
			print("Processing image: %s" %image_path)

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
			contours.sort(key = len, reverse = True)

			# center of image
			imX = im.shape[1] / 2
			imY = im.shape[0] / 2

			# for the 4 largest contours, compute the distance to center of image
			distance = -1
			n_contour = min(4, len(contours))	# there are not always 4 contours
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
			cv2.drawContours(mask, contours, goodContour, (255,255,255), cv2.FILLED)
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
			( _, thresh) = cv2.threshold(imgray, 15, 255, cv2.THRESH_BINARY)
			(contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

			# keep patch without holes
			mask = np.full(im.shape[:2], 255, dtype=np.uint8)
			for c in range(0, len(contours)):
				if (len(contours[c]) < 7000 and len(contours[c]) > 280):
					#print("contour: %d", len(contours[c]))
					cv2.drawContours(mask, contours, c, (0,0,0), cv2.FILLED)
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

			# copy read image
			image = np.copy(im)
			image_copy = np.copy(im)

			# get the actual inner list of hierarchy descriptions
			hierarchy = hierarchy[0]
			
			# initialize variables
			line = 0
			n_words = 0
			average_y = 0
			min_height = 20
			min_width = 20
			
			#loop over the found areas
			for component in zip(ctrs, hierarchy):
				# Get bounding box
				currentContour = component[0]
				currentHierarchy = component[1]
				x, y, w, h = cv2.boundingRect(currentContour)
				if currentHierarchy[3] < 0:  # these are the outermost parent components

					# Getting ROI
					roi = image[y:y+h, x:x+w]
					if h > min_height or w > min_width: # if image is not noise
						
						# determine line number line
						if h < 100:
							# normal segment
							if y + h/2 > average_y + 40 or y+h/2 < average_y - 40:
								 # next line
								line += 1
								average_y = y + h/2
								n_words = 0
							else:
								# same line
								average_y = (n_words * average_y + y + h/2) / (n_words + 1)
								n_words += 1
						
						# reset values
						old_y = y
						old_h = h
						old_x = x
						
						# Save segments
						cv2.imwrite(segmented_areas_path + '/' + folder + '/row ' + str(line) + ' col ' + str(x) + '.jpg', roi)
						
						# draw a rectangle around the segmented area
						if (h > 100):
							cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
						else:
							cv2.rectangle(image_copy, (x, y), (x + w, y + h), (90, 0, 255), 2)
						
						# draw number of contour
						cv2.putText(image_copy, str(line), cv2.boundingRect(currentContour)[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [125])
					else:
						cv2.imwrite(segmented_areas_path_trash + '/' + folder + '/row ' + str(y) + ' col ' + str(x) + '.jpg', roi)
			
			# save marked areas
			cv2.imwrite(marked_areas_path + '/' + scroll, image_copy)
	print("Completed scroll to img")

def convert_img_to_csv():
	print("Converting imgs to csv")
	Datadir = 'segmented_areas'
	training_data = []
	CATEGORIES = ["P21-Fg006-R-C01-R01-fused","P22-Fg008-R-C01-R01-fused","P106-Fg002-R-C01-R01-fused","P123-Fg001-R-C01-R01-fused","P123-Fg002-R-C01-R01-fused","P166-Fg002-R-C01-R01-fused","P166-Fg007-R-C01-R01-fused","P168-Fg016-R-C01-R01-fused","P172-Fg001-R-C01-R01-fused","P342-Fg001-R-C01-R01-fused","P344-Fg001-R-C01-R01-fused","P423-1-Fg002-R-C01-R01-fused","P423-1-Fg002-R-C02-R01-fused","P513-Fg001-R-C01-R01-fused","P564-Fg003-R-C01-R01-fused","P583-Fg002-R-C01-R01-fused","P583-Fg006-R-C01-R01-fused","P632-Fg001-R-C01-R01-fused","P632-Fg002-R-C01-R01-fused","P846-Fg001-R-C01-R01-fused"]
	#IMG_SIZE = 220
	def create_training_data():
		
		for category in CATEGORIES:
			path = os.path.join(Datadir,category)  
			for img in os.listdir(path): 
				img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
			
		
		for category in CATEGORIES:
			path = os.path.join(Datadir,category)  
			class_num = CATEGORIES.index(category)  # get the classification  (0 to 26).
			
			for img in tqdm(os.listdir(path)):  # iterate over each image per category
				
				try:
					IMG_SIZE = 28
					img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
					new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
					#print(new_array.shape)
					#training_data.append([new_array, class_num])  # add this to our training_data

					training_data.append([new_array, class_num])
				except Exception as e:  # in the interest in keeping the output clean...
					pass
				except OSError as e:
					print("OSErrroBad img most likely", e, os.path.join(path,img))
				except Exception as e:
					print("general exception", e, os.path.join(path,img))
		
		return training_data

	# testing if its getting printed
	training_data1 = create_training_data()

	X = []
	y = []

	for features,label in training_data1:
		X.append(features)
		y.append(label)

	X_train = []
	for img in X:
		new_img = np.reshape(img, 784)
		X_train.append(new_img)
		
	X_train = np.asarray(X_train)
	print(X_train.shape)
	np.savetxt("train.csv", X_train, delimiter=",")
	print("Completed img to csv")

# convert scroll img to character imgs
convert_scroll_to_imgs()
# convert imgs to csv
convert_img_to_csv()
# run model
