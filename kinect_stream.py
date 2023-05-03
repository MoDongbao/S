# -*- coding: UTF-8 -*- #
########################################################
#                                                      #
# Copyright (c) 2023, Mo Dongbao. All Rights Reserved. #
#                                                      #
########################################################

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = FrameType.Color
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.startStreams(rgb=True, depth=False)

def stream():
    frames = listener.waitForNewFrame()
    color = frames["color"]
    img = cv2.resize(cv2.flip(cv2.cvtColor(color.asarray(), cv2.COLOR_BGR2RGB), 1), (int(1920 / 3), int(1080 / 3)))
    listener.release(frames)
    if img is None:
        pass
    else:
        return img

def stream_close():
	device.stop()
	device.close()
	sys.exit(0)

