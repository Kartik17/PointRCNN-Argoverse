import numpy as np
import argparse
import sys
import os
import json
from scipy.spatial.transform import Rotation
from bbox3d import BBox3D
from metrics import iou_3d

def build_argo_object_list(label_data, classes_included = ['VEHICLE']):
	label_objects_list = []
	for data in label_data:
		if data['label_class'] in classes_included:
			cx, cy, cz = data['center']['x'], data['center']['y'], data['center']['z']
			l, w, h = data['length'], data['width'], data['height']
			rw, rx, ry, rz = data['rotation']['w'], data['rotation']['x'],data['rotation']['y'],data['rotation']['z']
			temp = BBox3D(cx, cy, cz,
        	         length=l, width=w, height=h,
        	         rw=rw, rx=rx, ry=ry, rz=rz, is_center=True)
			label_objects_list.append(temp)

	return label_objects_list

def evaluate_labels(prediction_data,label_data):
	TP = 0.0
	total_gt = len(label_data)
	visited = [False for i in range(total_gt)]
	
	for predict_box in prediction_data:
		for i,label_box in enumerate(label_data):
			IOU_3D = iou_3d(predict_box, label_box)
			
			if(IOU_3D > 0.7) and visited[i] == False:
				visited[i] = True
				TP += 1.0


				
	FP = len(prediction_data) - TP
	return TP, FP, total_gt

if __name__ == '__main__':	

	parser = argparse.ArgumentParser()
	parser.add_argument('--label_path', type = str,help = "Path of the labels")
	parser.add_argument('--prediction_path', type = str,help = "Path of the prediction")

	args = parser.parse_args()

	if not os.path.isdir(args.label_path):
		print("label directory doesn't exist")

	if not os.path.isdir(args.prediction_path):
		print("prediction directory doesn't exist")

	prediction_files = sorted(os.listdir(args.prediction_path))
	label_files = sorted(os.listdir(args.label_path))
	
	assert len(prediction_files) == len(label_files)
	total_frames = len(label_files)

	ap_list = []
	total_labels = 0
	total_predicted = 0
	for i in range(total_frames):
		prediction_content =  open(os.path.join(args.prediction_path,prediction_files[i]))
		label_content = open(os.path.join(args.label_path,label_files[i]))

		prediction_data = build_argo_object_list(json.load(prediction_content))
		label_data = build_argo_object_list(json.load(label_content))

		TP,FP, Total_GT = evaluate_labels(prediction_data,label_data)
		ap_list.append(TP/(TP+FP))
		total_predicted += TP
		total_labels += Total_GT


	MAP = np.mean(ap_list)
	recall = (total_predicted/total_labels)
	print(np.round(MAP,2), np.round(recall,2))
		