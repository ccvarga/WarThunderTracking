import time
import windows_capture as wc
import cv2 as cv
import onnxruntime
import numpy as np
from ultralytics import YOLO
from ultralytics import p

#session = onnxruntime.InferenceSession('bestLarge.onnx')

session = YOLO('bestLarge.onnx')

def custom_frame_handler(frame: wc.Frame, capture_control: wc.CaptureControl):
    start = time.perf_counter()

    screenShot = cv.cvtColor(frame.frame_buffer, cv.COLOR_BGRA2BGR)

    result = session.predict(screenShot,imgsz=[1216,1920])

    result = session.plot()

    cv.imshow('tracking', result)

    stop = time.perf_counter()
    print("Writing Duration:", (stop - start) * 1000, "ms")

def custom_closed_handler() -> None:
    pass





wincap = wc.WindowsCapture()
wincap.frame_handler = custom_frame_handler
wincap.closed_handler = custom_closed_handler
wincap.start()

wincap.stop()
