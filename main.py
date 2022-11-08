from tokenize import Number
import cv2
import os
from PIL import Image, ImageDraw
import torch
from facenet_pytorch import MTCNN
from dotenv import load_dotenv
import dotenv
import sys
from retinaface import RetinaFace

# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'


class Detector:
    def __init__(self, name) -> None:
        self.name = name
        if name == "mtcnn":
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model = MTCNN(keep_all=True, device=device)
        elif name == "cascade":
            self.model = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        else: 
            self.model = RetinaFace
        print("detector: ", self.name)
    def detect(self, image):
        if self.name == "mtcnn":
            boxes, scores = self.model.detect(image)
        elif self.name == "cascade":
            pass
        else: 
            resp = self.model.detect_faces(image)
            boxes = []
            scores = []
            for i in resp:
                boxes.append(resp[i]["facial_area"])
                scores.append(resp[i]["score"])
        return boxes, scores


def collect_face_images(cam_name, model):

    RTSP_URL = get_RTSP_URL(cam_name)

    cap = cv2.VideoCapture(0)
    detector = Detector(model)

    if not cap.isOpened():
        print('Cannot open RTSP stream')
        exit(-1)

    frame_number = 0
    number_image = (count_file(cam_name) - 1)/2

    while True:
        _, frame = cap.read()
        draw_img = frame
        if frame_number % 60 == 0:
            try:
                boxes, confident = detector.detect(frame)
                print(confident)

                if (type(boxes) != type(None)):

                    # cv2.imwrite('./data/{}/{}.png'.format(cam_name, str(number_image)), frame)
                    # cv2.imwrite('./data/{}/{}.box.png'.format(cam_name, str(number_image)), draw_boxes(frame, boxes))

                    print("Saved image [{}] with {} boxes".format(number_image, len(boxes)))
                    number_image += 1
                    frame_number -= 1
                    draw_img = draw_boxes(frame, boxes)
                else:
                    print("Don't have any face on image", frame_number)
            except:
                print("Error: can't detect this frame")
            

        cv2.imshow('RTSP stream', draw_img)
        frame_number += 1
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def detect_face(image, model):
    detecter = Detector(model)
    boxes, _ = detecter.detect(image)
    if (type(boxes) != type(None)):
        image = draw_boxes(image, boxes)
    while True:
        cv2.imshow("test cam 1", image)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def main():

    cam_name = sys.argv[1]
    print("running in ", cam_name)

    collect_face_images(cam_name, 'retina')
    # img = cv2.imread("cam_1_background.png")
    # detect_face(img, 'mtcnn')
    
if __name__ == "__main__":
    main()

def init_detector(type):
    if type == "mtcnn":
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))
        detector = MTCNN(keep_all=True, device=device)
    elif type == "cascade":
        detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    elif type == "retina": 
        detector = RetinaFace
    else: 
        print("this detector not found")
    return detector

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
        for box in boxes:
            box = [int(i) for i in box]
            [x1, y1, x2, y2] = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return img

def get_RTSP_URL(cam_name):
    dotenv_file = "./data/{}/.env".format(cam_name)
    load_dotenv(dotenv_file)
    RTSP_URL = os.getenv('RTSP_URL')
    print("RTSP_URL: ", RTSP_URL)
    return RTSP_URL