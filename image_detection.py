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

edge.Log.set_level(logging.DEBUG)

try:
    # _model_dir = sys.argv[1]
    # _test_file = sys.argv[2]
    _model_dir = "./RES/"
    #_test_file = sys.argv[1]
except Exception as e:
    print("Usage: python3 image_detection.py")
    exit(-1)


pred = edge.Program()
pred.set_auth_license_key("5B8E-3839-3581-1514")

# jetson sdk支持一些特殊的参数配置，具体支持哪些参数配置可以通过 help(edge.EdgekitGeneralConfig) 和 help(edge.JetsonConfig)查看
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

def detect(img):
	height, width, _ = img.shape
	result = pred.infer_image(img, threshold=None)
	for det in result:
		print("{}, {}, p:{}".format(det.index, det.label, det.prob), end="")
		# 物体检测有一个框
		if pred.model_type in [
				edge.c.ModelType.ObjectDetection, edge.c.ModelType.FaceDetection, edge.c.ModelType.ImageSegmentation
		]:
			print(", coordinate: {}, {}, {}, {}".format(det.x1, det.y1, det.x2, det.y2), end="")
			cv2.rectangle(img, (int(width * det.x1), int(height * det.y1)),
						  (int(width * det.x2), int(height * det.y2)), (0, 0, 255), 2)
			cv2.putText(img=img,
						text=det.label + " " + str(det.prob),
						org=(int(width * det.x1), int(height * det.y1)),
						fontFace=cv2.FONT_HERSHEY_PLAIN,
						fontScale=1,
						color=(0, 0, 255),
						thickness=1)

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
            pred.close()
            break
