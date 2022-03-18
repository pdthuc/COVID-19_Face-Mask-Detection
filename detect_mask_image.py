# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="8"

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet_v2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet_50
from classification_models.tfkeras  import Classifiers 

# for tensorflow.keras
# from classification_models.tfkeras import Classifiers
ResNet34, preprocess_input_resnet_34 = Classifiers.get('resnet34')


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from tqdm import tqdm
import time

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score

import matplotlib.pyplot as plt
import numpy as np

def mask_image():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="mask_detector",
		help="path to trained face mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	model = load_model(args["model"])
	root_dir = args["image"]

	# count_total = 0
	# count_undetect = 0
	# count_detect = 0
	# count_true_mask = 0
	# count_true_noNask = 0

	true = []
	pred = []
	labels = ["NoMask", "Mask"]
	incorrect_imgs = []

	for directory, subdirectories, files in os.walk(root_dir):
		for file in tqdm(files):
			# print(os.path.join(directory, file))
			# print(" - ", directory.split(os.path.sep)[-1])
			# print(" - ", file)
		
			image_path = os.path.join(directory, file)
			image_label = directory.split(os.path.sep)[-1]
			if image_label == "Mask":
				true.append(1) 
			else:
				true.append(0)

			# for test
			# image_label = "NoMask"


			image = cv2.imread(image_path)
			# count_total += 1
			
			# (h, w) = image.shape[:2]

			# # construct a blob from the image
			# blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
			# 	(104.0, 177.0, 123.0))
			# # pass the blob through the network and obtain the face detections
			# net.setInput(blob)
			# detections = net.forward()

			
				# # loop over the detections
				# for i in range(0, detections.shape[2]):
				# 	# extract the confidence (i.e., probability) associated with
				# 	# the detection
				# 	confidence = detections[0, 0, i, 2]

				# 	# filter out weak detections by ensuring the confidence is
				# 	# greater than the minimum confidence
				# 	if confidence > args["confidence"]:
				# 		# compute the (x, y)-coordinates of the bounding box for
				# 		# the object
				# 		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				# 		(startX, startY, endX, endY) = box.astype("int")

				# 		# ensure the bounding boxes fall within the dimensions of
				# 		# the frame
				# 		(startX, startY) = (max(0, startX), max(0, startY))
				# 		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# 		# extract the face ROI, convert it from BGR to RGB channel
				# 		# ordering, resize it to 224x224, and preprocess it
				# 		face = image[startY:endY, startX:endX]
			face = image
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
            
			if "resnet_50" in args["model"]:
				face = preprocess_input_resnet_50(face)
			elif "resnet_34" in args["model"]:
				face = preprocess_input_resnet_34(face)
			elif "mobilenet" in args["model"]:
				face = preprocess_input_mobilenet_v2(face)    
                
			face = np.expand_dims(face, axis=0)

			# pass the face through the model to determine if the face
			# has a mask or not
			(mask, withoutMask) = model.predict(face)[0]

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "NoMask"
			# color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			# print("\n{} - {}".format(image_label, label))
			if mask > withoutMask:
				pred.append(1) 
			else:
				pred.append(0)

			if image_label != label:
				incorrect_imgs.append(image_path)

# 				cv2.imshow("False", image)
# 				cv2.waitKey(0)
				
						# cv2.putText(image, label, (startX, startY - 10),
						# cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
						# cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
						# cv2.imshow("False", image)
						# cv2.waitKey(0)
						# print("{} - {}".format(image_label, label))
						# else:
						# 	cv2.putText(image, label, (startX, startY - 10),
						# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
						# 	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
						# 	cv2.imshow("False", image)
						# 	cv2.waitKey(0)
						# 	print("{} - {}".format(image_label, label))						

						# include the probability in the label
						# label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
	cm = confusion_matrix(true, pred)
# 	print("test: ", true)
# 	print("predict: ", pred)

	print(" - Accuracy: ", accuracy_score(true, pred))
	print(" - Recall: ", recall_score(true, pred, average=None))
	print(" - Precision: ", precision_score(true, pred, average=None))
    
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

	disp.plot(cmap=plt.cm.Blues)
     
	if "resnet_50" in args["model"]:
		plt.savefig('output_confusionMatrix_png/resnet_50.png')
		textfile = open("incorrect_label/resnet_50_incorrect_label.txt", "w")
		for element in incorrect_imgs:
			textfile.write(element + "\n")
		textfile.close()
	elif "resnet_34" in args["model"]:
		plt.savefig('output_confusionMatrix_png/resnet_34.png')
		textfile = open("incorrect_label/resnet_34_incorrect_label.txt", "w")
		for element in incorrect_imgs:
			textfile.write(element + "\n")
		textfile.close()
	elif "mobilenet" in args["model"]:
		plt.savefig('output_confusionMatrix_png/mobilenet_v2.png')
		textfile = open("incorrect_label/mobilenet_v2_incorrect_label.txt", "w")
		for element in incorrect_imgs:
			textfile.write(element + "\n")
		textfile.close()

if __name__ == "__main__":
	mask_image()

