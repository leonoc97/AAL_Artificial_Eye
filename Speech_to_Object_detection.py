import keyboard
import speech_recognition as sr
from ultralytics import YOLO
import cv2

def listen_for_keyword():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for keyword...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None

def process_text(text):
    keywords = ["banana", "apple", "cell phone", "cup", "bottle"]  # Note: "cell phone" instead of "cellphone"
    detected_keywords = [keyword for keyword in keywords if keyword in text]
    return detected_keywords


# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
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
img = None  # Initialize img outside the loop
detected_objects = [] # List to store detected objects for tracking
while True:
    keyboard.wait("space")
    detected_text = listen_for_keyword()

    if detected_text is not None:
        print(f"Detected: {detected_text}")
        detected_keywords = process_text(detected_text)

        if detected_keywords:
            print(f"Detected keywords: {', '.join(detected_keywords)}")

            success, img = cap.read()
            results = model(img, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    if class_name in detected_keywords:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detected_objects.append((class_name, (x1, y1, x2, y2)))

        else:
            print("No keywords detected.")
    else:
        print("No speech detected.")



    if img is not None:
        for class_name, (x1, y1, x2, y2) in detected_objects:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    