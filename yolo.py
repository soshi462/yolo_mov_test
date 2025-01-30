import cv2
from ultralytics import YOLO
import torch
import threading

# ??????
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# YOLO????????
model = YOLO("yolo11n.pt")
model.to(device)

# ??????
camera = cv2.VideoCapture(0)  # ??????? 0: ?????
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# ?????????
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ?????????
frame = None
lock = threading.Lock()
stop_flag = False


def camera_reader():
    global frame, stop_flag
    while not stop_flag:
        ret, temp_frame = camera.read()
        if ret:
            with lock:
                frame = temp_frame


# ?????????????????
thread = threading.Thread(target=camera_reader)
thread.start()

print("Press 'q' to quit.")

try:
    while True:
        # ?????????
        with lock:
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ??
        results = model(rgb_frame, imgsz=320, conf=0.5)

        # ??????: YOLO?plot?????
        annotated_frame = results[0].plot()

        # ?????
        cv2.imshow("YOLO Detection", annotated_frame)

        # 'q'???????
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break

finally:
    # ???????????
    stop_flag = True
    thread.join()
    camera.release()
    cv2.destroyAllWindows()