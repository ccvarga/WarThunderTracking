import time
import windows_capture as wc
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from PIL import Image

#session = onnxruntime.InferenceSession('bestLarge.onnx')

session = YOLO('bestLarge.onnx')

def custom_frame_handler(frame: wc.Frame, capture_control: wc.CaptureControl):
    start = time.perf_counter()

    screenshot = cv.cvtColor(frame.frame_buffer, cv.COLOR_BGRA2BGR)

    res = session.predict(screenshot,imgsz=[1216,1920])[0]

    res = res.plot(line_width=1)
    res = res[:, :, ::-1]
    #res = Image.fromarray(res)


    cv.imshow('tracking', res)

    stop = time.perf_counter()
    print("Writing Duration:", (stop - start) * 1000, "ms")
    if cv.waitKey(1) == ord('q'):
        capture_control.stop()
        cv.destroyAllWindows

def custom_closed_handler() -> None:
    pass





wincap = wc.WindowsCapture()
wincap.frame_handler = custom_frame_handler
wincap.closed_handler = custom_closed_handler
wincap.start()

wincap.stop()
