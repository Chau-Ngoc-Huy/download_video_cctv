import cv2
import os


RTSP_URL = 'http://127.0.0.1:8080'

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# from cctv
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
# from local cam
# cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

while True:
    _, frame = cap.read()
    cv2.imshow('RTSP stream', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()