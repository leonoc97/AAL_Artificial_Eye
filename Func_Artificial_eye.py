import speech_recognition as sr
import cv2
import math 

def process_text(text):
    keywords = ["banana", "apple", "cell phone", "cup", "bottle"] 
    detected_keywords = [keyword for keyword in keywords if keyword in text]
    return detected_keywords


def listen_for_keyword():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for keyword...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None
    

def plot_box(results, img, target_classes, classNames):
    # coordinates
    center_x = 0
    center_y = 0
    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            if classNames[cls] in target_classes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                center_x = math.floor((x2+x1)/2)
                center_y = math.floor((y2+y1)/2)
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img,(center_x,center_y),10,(0,0,255),10)
                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            return img, center_x, center_y