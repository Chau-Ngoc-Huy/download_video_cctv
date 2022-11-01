from tokenize import Number
import cv2
import os
from PIL import Image, ImageDraw
import torch
from facenet_pytorch import MTCNN
from dotenv import load_dotenv


def init_detecter():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(keep_all=True, device=device)
    return mtcnn

def draw_boxes(img, boxes):
    if type(boxes) == type(None):
        print("don't have any box to draw")
    else:
        for [x1, y1, x2, y2] in boxes:
            # print(box)
            # print(x1, y1, x2, y2)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return img

def main():
    RTSP_URL = 'http://127.0.0.1:8080'
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

    load_dotenv()
    number_image = os.getenv('NUMBER_IMAGE')

    # from cctv
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    # from local cam
    # cap = cv2.VideoCapture(0)

    detecter = init_detecter()


    if not cap.isOpened():
        print('Cannot open RTSP stream')
        exit(-1)

    while True:
        _, frame = cap.read()
        boxes, _ = detecter.detect(frame)

        # frame_draw = draw_boxes(frame, boxes)

        # cv2.imshow('RTSP stream', frame_draw)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()