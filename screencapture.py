import mmap
import time
import windows_capture as wc
import cv2 as cv

print("Opening camera...")

mm = None

def custom_frame_handler(frame: wc.Frame, capture_control: wc.CaptureControl):
    global mm
    
    to_bgr = cv.cvtColor(frame.frame_buffer,cv.COLOR_BGRA2BGR)

    if mm is None:
        mm = mmap.mmap(-1, to_bgr.size)

    start = time.perf_counter()
    buf = to_bgr.tobytes()
    mm.seek(0)
    mm.write(buf)
    mm.flush()
    stop = time.perf_counter()
    print(to_bgr.size)
    print("Writing Duration:", (stop - start) * 1000, "ms")

def custom_closed_handler() -> None:
    pass

wincap = wc.WindowsCapture()

wincap.frame_handler = custom_frame_handler
wincap.closed_handler = custom_closed_handler

wincap.start()

wincap.stop()
mm.close()