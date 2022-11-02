from tokenize import Number
import cv2
import os
from PIL import Image, ImageDraw
import torch
from facenet_pytorch import MTCNN
from dotenv import load_dotenv
import dotenv
import sys

# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

def init_detecter():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(keep_all=True, device=device)
    return mtcnn

def count_file(cam_name):

    dir = "./data/" + cam_name
    count = 0
    for path in os.listdir(dir):
        count += 1
    return count

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
    number_image = count_file(cam_name) - 1
    RTSP_URL = os.getenv('RTSP_URL')
    print("RTSP_URL: ", RTSP_URL)

    cap = cv2.VideoCapture(RTSP_URL)
    detecter = init_detecter()

    if not cap.isOpened():
        print('Cannot open RTSP stream')
        exit(-1)
    frame_number = 0

    while True:
        _, frame = cap.read()
        if frame_number % 30 == 0:
            boxes, _ = detecter.detect(frame)
            if (type(boxes) != type(None)):
                cv2.imwrite('./data/{}/{}.png'.format(cam_name, str(number_image)), frame)

                print("Saved image [{}] with {} boxes".format(number_image, len(boxes)))
                number_image += 1
                frame_number -= 1
            else:
                print("Don't have any face on image", frame_number)

            # draw_img = draw_boxes(frame, boxes)

        # cv2.imshow('RTSP stream', draw_img)
        frame_number += 1
        if cv2.waitKey(1) == 27:
            break
    
    dotenv.set_key(dotenv_file, "NUMBER_IMAGE", 20)
    cap.release()
    cv2.destroyAllWindows()

def detect_face(image):
    detecter = init_detecter()
    boxes, _ = detecter.detect(image)
    if (type(boxes) != None):
        drawed_img = draw_boxes(image, boxes)
    cv2.imshow("test cam 1", drawed_img)


def main():

    cam_name = sys.argv[1]
    print("running in ", cam_name)

    collect_face_images(cam_name)
    # img = cv2.imread("cam_1_background.png")
    # detect_face(img)
    
if __name__ == "__main__":
    main()
