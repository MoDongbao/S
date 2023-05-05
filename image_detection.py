# -*- coding: UTF-8 -*- #
########################################################
#                                                      #
# Copyright (c) 2023, Mo Dongbao. All Rights Reserved. #
#                                                      #
########################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import multiprocessing
import sys

import cv2
import numpy as np
import BaiduAI.EasyEdge as edge
import logging

from kinect_stream import stream, stream_close
import time

# edge.Log.set_level(logging.DEBUG)

try:
    _model_dir = "./RES/"
except Exception as e:
    print("\n{model_dir} Error! Please check it!\n")
    exit(-1)


pred = edge.Program()
pred.set_auth_license_key("5B8E-3839-3581-1514")

# Jetson SDK 支持一些特殊参数配置，通过 help(edge.EdgekitGeneralConfig) 和 help(edge.JetsonConfig)查看
edgekit_specific_config = {
	edge.EdgekitGeneralConfig.PREDICTOR_KEY_DEVICE_ID: 0,
	edge.JetsonConfig.PREDICTOR_KEY_GTURBO_MAX_CONCURRENCY: 1,
	edge.JetsonConfig.PREDICTOR_KEY_GTURBO_FP16: False,
	edge.JetsonConfig.PREDICTOR_KEY_GTURBO_COMPILE_LEVEL: 1
}
pred.init(model_dir=_model_dir,
		  device=edge.Device.Jetson,
		  engine=edge.Engine.EDGEKIT_JETSON,
		  thread_num=4,
		  **edgekit_specific_config)

last_timestamp = 0
this_timestamp = 1


def detect(img):
	global this_timestamp, last_timestamp
	height, width, _ = img.shape
	result = pred.infer_image(img, threshold=None)
	for det in result:
		# print("{}, {}, p:{}".format(det.index, det.label, det.prob), end=", ")
		# 物体检测框
		if pred.model_type in [
				edge.c.ModelType.ObjectDetection, edge.c.ModelType.FaceDetection, edge.c.ModelType.ImageSegmentation
		]:
			# print("coordinate: {}, {}, {}, {}".format(det.x1, det.y1, det.x2, det.y2), end="\n")
			cv2.rectangle(img, (int(width * det.x1), int(height * det.y1)),
					  (int(width * det.x2), int(height * det.y2)), (0, 0, 255), 2)
			cv2.putText(img=img,
					text=det.label + " " + str(round(det.prob, 3)),
					org=(int(width * det.x1 - 4), int(height * det.y1 - 2)),
					fontFace=cv2.FONT_HERSHEY_PLAIN,
					fontScale=2,
					color=(0, 255, 0),
					thickness=2)
	this_timestamp = time.time()
	cv2.putText(img=img,
			text="FPS" + " " + str(round(1 / (this_timestamp-last_timestamp), 1)),
			org=(0, height),
			fontFace=cv2.FONT_HERSHEY_PLAIN,
			fontScale=2,
			color=(255, 255, 255),
			thickness=2)
	last_timestamp = this_timestamp
	if pred.model_type in [edge.c.ModelType.ObjectDetection, edge.c.ModelType.FaceDetection, edge.c.ModelType.ImageSegmentation]:
		return img

if __name__ == "__main__":
    while True:
        img = stream()
        cv2.imshow("stream", img)
        img = detect(img)
        cv2.imshow('detect', img)
        key = cv2.waitKey(delay = 1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            stream_close()
            pred.close()
            break

