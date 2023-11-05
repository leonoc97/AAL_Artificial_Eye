from ultralytics import YOLO
import cv2
import math 
import keyboard
import speech_recognition as sr
import Func_Artificial_eye

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# initialize model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]  

# initialize variables 
target_classes = [  ]
detected_text = []

while True:
    # only listens when space is pressed
    if keyboard.is_pressed("space"):
        detected_text = Func_Artificial_eye.listen_for_keyword()

    if detected_text is not None:
        print(f"Detected: {detected_text}")
        # if you want to listen to other keywords, change them in process_text
        target_classes = Func_Artificial_eye.process_text (detected_text)

        if target_classes:
            print(f"Detected keywords: {', '.join(target_classes)}")

    # object detection 
    success, img = cap.read()
    results = model(img, stream=True);

    # plot boxes around object
    # coordinates of detected oject
    img, center_x, center_y = Func_Artificial_eye.plot_box(results,img,target_classes,classNames)   #  runs into error if no objects are detected --> check if results it not empty

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
