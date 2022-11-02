from tokenize import Number
import cv2
import os
from PIL import Image, ImageDraw
import torch
from facenet_pytorch import MTCNN
from dotenv import load_dotenv
import dotenv

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

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

def collect_face_images(cam_name):

    dotenv_file = "./data/{}/.env".format(cam_name)
    load_dotenv(dotenv_file)
    number_image = int(os.getenv('NUMBER_IMAGE'))
    RTSP_URL = os.getenv('RTSP_URL')

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    detecter = init_detecter()

    if not cap.isOpened():
        print('Cannot open RTSP stream')
        exit(-1)

    while True:
        _, frame = cap.read()
        boxes, _ = detecter.detect(frame)
        if (type(boxes) != None):
            cv2.imwrite('./data/{}/{}.png'.format(cam_name, str(number_image)), frame)
            number_image += 1

            # draw_img = draw_boxes(frame, boxes)

        # cv2.imshow('RTSP stream', draw_img)
        if cv2.waitKey(1) == 27:
            break
    
    dotenv.set_key(dotenv_file, "NUMBER_IMAGE", number_image)
    cap.release()
    cv2.destroyAllWindows()

def main():

    collect_face_images("cam_1")
    collect_face_images("cam_2")

    
if __name__ == "__main__":
    main()