import cv2
import numpy as np
import requests
import time

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]


WRITE_API_KEY = "4GCWNDENM6E0AHGL"
CHANNEL_ID = 2528678


def detect_objects(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, boxes, class_ids  

def main():
    cap = cv2.VideoCapture(0)
    object_detected = False
    while True:
        ret, frame = cap.read()
        indexes, boxes, class_ids = detect_objects(frame)  

        if len(indexes) > 0:
            object_detected = True
        else:
            object_detected = False

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

             
                x_center = x + w / 2
                servo_angle = int(np.interp(x_center, [0, frame.shape[1]], [0, 180]))

                
                params = {
                    "api_key": WRITE_API_KEY,
                    "field1": servo_angle
                }
                response = requests.post(f"https://api.thingspeak.com/update?api_key={WRITE_API_KEY}&field1={servo_angle}")
                print(f"Servo Angle: {servo_angle}")


        if object_detected:
            requests.post(f"https://api.thingspeak.com/update?api_key={WRITE_API_KEY}&field2=1")
            print("Object detected, Signal sent: 1")
        else:
            requests.post(f"https://api.thingspeak.com/update?api_key={WRITE_API_KEY}&field2=0")
            print("No object detected, Signal sent: 0")

        cv2.imshow("Farm Field Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()