
import cv2
import numpy as np 
import argparse
import time
import collections

hash={0:'-',1:2,2:0,3:5,4:9,5:1,6:8,7:7,8:4,9:6,10:3}
Order_Threshold=4
file = open('/content/drive/MyDrive/predictions/predicted.txt','w')

def reorder_box(box,class_id):
	# print(class_id)
	hash={}

	for i in range(len(box)):
		hash[box[i]]=class_id[i]


	hash = collections.OrderedDict(sorted(hash.items()))
	num=[]
	for i in hash.values():
		num.append(i)
	
	NUMBER=""
	for i in range(len(num)):
		if len(num)>=Order_Threshold:
			if len(num)==4:
				if i==2:
					NUMBER=NUMBER+'.'+str(num[i])
				else:
					NUMBER=NUMBER+str(num[i])
		
			if len(num)==5:
				if i==3:
					NUMBER=NUMBER+'.'+str(num[i])
				else:
					NUMBER=NUMBER+str(num[i])
		else:
			print("Prediction Size is less than 3")
			break
	 		
	if len(NUMBER)>3 and NUMBER[0]!="-"  :
		NUMBER="-"+NUMBER
	
	return NUMBER

def load_yolo():
	net = cv2.dnn.readNet("/content/darknet/yolov3_final.weights", "/content/darknet/cfg/yolov3.cfg")
	classes = []
	with open("/content/darknet/data/obj.names", "r") as f:
		classes = [line.strip() for line in f.readlines()] 
	
	output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
	ordered_box=[]
	ordered_class=[]

	boxes = []
	confs = []
	class_ids = []
	sc=0.3
	passes=6
	check=0
	while len(class_ids)!=5:
		check=check+1
		if(check==passes):
			break
		for output in outputs:
			for detect in output:
				
				scores = detect[5:]

				class_id = np.argmax(scores)

				conf = scores[class_id]
				if conf > sc:
					center_x = int(detect[0] * width)
					center_y = int(detect[1] * height)
					w = int(detect[2] * width)
					h = int(detect[3] * height)
					x = int(center_x - w/2)
					y = int(center_y - h / 2)
					boxes.append([x, y, w, h])
					
					confs.append(float(conf))
					class_ids.append(class_id)
					ordered_box.append(x)
					ordered_class.append(hash[class_id])
		if len(class_ids)==5:
			break
		else:
			class_ids = []
			boxes = []
			confs = []
			ordered_box=[]
			ordered_class=[]
		sc=sc-0.025
	NUMBER=reorder_box(ordered_box,ordered_class)
	return boxes, confs, class_ids,NUMBER

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	ordered_box=[]
	ordered_class=[]

	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = (255,0,0)
			if label=="ifrsuit":
				color = (0,255,0)
			else:
				color = (0,0,255)
			ordered_box.append(x)
			ordered_class.append(hash[class_ids[i]])
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 2, color, 2)
	 		
	cv2.imshow("Labelled Image",img)
	return img
    
def OCR(img):
	model, classes, colors, output_layers = load_yolo()
	height, width,channels = img.shape
	blob, outputs = detect_objects(img, model, output_layers)
	boxes, confs, class_ids,NUMBER = get_box_dimensions(outputs, height, width)
	print("Detected No is : ",NUMBER)
	if(NUMBER==""):
		file.write("None"+"\n")
	else:
		file.write(NUMBER.strip()+"\n")

	finalImg = draw_labels(boxes, confs, colors, class_ids, classes, img)
	cv2.imwrite("/content/without/example"+str(i)+".png",finalImg)
 

if __name__=="__main__":
	for i in range(1,41):
		img=cv2.imread('/content/drive/MyDrive/Reader Images/Fiber'+str(i)+'.jpg')
		OCR(img)
	file.close()






