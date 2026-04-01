import cv2
import numpy as np
FRAME_COUNT =5
IMG_SIZE = 640
STEP= 5

def get_class_ids(name):
  if'NonViolence' == name:
    return 1
  else:
    return 0
def get_frames(path, nframes=FRAME_COUNT, size=(IMG_SIZE, IMG_SIZE),  frame_step = STEP):
    frames = []
    cap = cv2.VideoCapture(path)


    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get the total number of frames in the video
    # print(total_frames)

    for _ in range(nframes):
        # ret is a boolean indicating whether read was successful,
        # frame is the image itself
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros_like(frames[0]))
            continue

        # Process the current frame (resize, convert, normalize)
        # frame = cv2.resize(frame, size)
        # # frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
        # frame = frame / 255.0
        frames.append(frame)

        # Skip the specified number of frames
        for _ in range(frame_step):
            cap.grab()  # Advance to the next frame without reading it

    cap.release()
    return frames