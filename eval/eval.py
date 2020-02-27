import numpy as np
import argparse
import sys
import os
import json
from scipy.spatial.transform import Rotation
from bbox3d import BBox3D
from metrics import iou_3d
import matplotlib.pyplot as plt
from numba import jit

PC_AREA_SCOPE = [[-40, 40], [-3,   3], [-60, 60]]

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


def valid_box_filter(box3d_list):
	valid_gt_list = []

	for box3d in box3d_list:
		if( ( (box3d.cx <= PC_AREA_SCOPE[2][1]) & (box3d.cx >= PC_AREA_SCOPE[2][0]) )\
		  & ( (box3d.cy <= PC_AREA_SCOPE[0][1]) & (box3d.cy >= PC_AREA_SCOPE[0][0]) ) \
		  & ( (box3d.cz <= PC_AREA_SCOPE[1][1]) & (box3d.cz >= PC_AREA_SCOPE[1][0])) ):
			valid_gt_list.append(box3d)
	print("Total_box:{}, Valid_Box:{}".format(len(box3d_list), len(valid_gt_list)))
	return valid_gt_list

def evaluate_labels(prediction_data,label_data, IOU_threshold = 0.7):
	if (IOU_threshold > 1) & (IOU_threshold <= 0):
		raise('IOU Threshold should be between 0 and 1')

	TP = 0.0
	valid_gt_list = valid_box_filter(label_data)
	total_gt = len(valid_gt_list)
	visited = [False for i in range(total_gt)]
	
	for predict_box in prediction_data:
		for i,label_box in enumerate(valid_gt_list):
			IOU_3D = iou_3d(predict_box, label_box)
			
			if(IOU_3D > IOU_threshold) and (visited[i] == False):
				visited[i] = True
				TP += 1.0

	FP = len(prediction_data) - TP # As only VEHICLE class is detected
	print("TP:{}, FP:{}".format(TP,FP))
	return TP, FP, total_gt

def MAP_graph(prediction_path, label_path):


	

	precision_list = np.zeros(9)
	recall_list = np.zeros(9)
	iou = 0.1

	for i in range(8):
		AP, recall = single_iou_evaluation(prediction_path, label_path, iou)
		iou += 0.1
		print('IOU :{}, AP: {}, and Recall: {}'.format(iou, AP, recall))
		precision_list[i] = AP
		recall_list[i] = recall

	plt.plot(recall_list, precision_list)
	plt.show()

def single_iou_evaluation(prediction_path, label_path, iou=0.7):



	prediction_files = sorted(os.listdir(prediction_path))
	label_files = sorted(os.listdir(label_path))

	total_frames = len(prediction_files)
	ap_list = np.zeros(total_frames)
	total_labels = 0
	total_predicted = 0


	for i in range(total_frames):
		prediction_content =  open(os.path.join(args.prediction_path,prediction_files[i]))
		label_content = open(os.path.join(args.label_path,label_files[i]))

		prediction_data = build_argo_object_list(json.load(prediction_content))
		label_data = build_argo_object_list(json.load(label_content))

		TP,FP, Total_GT = evaluate_labels(prediction_data,label_data,iou)
		ap_list[i] = TP/(TP+FP)
		total_predicted += TP
		total_labels += Total_GT

	AP = np.mean(ap_list)
	recall = (total_predicted/total_labels)
	return AP, recall


if __name__ == '__main__':	

	parser = argparse.ArgumentParser()
	parser.add_argument('--label_path', type = str,help = "Path of the labels")
	parser.add_argument('--prediction_path', type = str,help = "Path of the prediction")
	parser.add_argument('--iou', type =float, help = "IOU Threshold")

	args = parser.parse_args()

	if not os.path.isdir(args.label_path):
		print("label directory doesn't exist")

	if not os.path.isdir(args.prediction_path):
		print("prediction directory doesn't exist")

	
	#MAP_graph(args.prediction_path, args.label_path)
	AP, recall = single_iou_evaluation(args.prediction_path, args.label_path, args.iou)
	print('Avergae Precision:{} and Recall:{}'.format(AP, recall))
