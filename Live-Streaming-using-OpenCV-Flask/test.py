from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import random

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading camera

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    print("I'm in video feed")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 인식된 사물의 라벨을 저장하는 리스트
save = []
index = 0

@app.route('/save_feed')
def save_feed():
    print("I'm in save feed")

    global index
    page_selection = ''

    print(index)

    if index == 0:
        if 'cell phone' in save:
            page_selection = 'goodResult.html'
            save.clear()
        else:
            page_selection = 'bad_index.html'
            save.clear()
    elif index == 1:
        if 'bowl' in save:
            page_selection = 'goodResult.html'
            save.clear()
        else:
            page_selection = 'bad_index.html'
            save.clear()
    elif index == 2:
        if 'cup' in save:
            page_selection = 'goodResult.html'
            save.clear()
        else:
            page_selection = 'bad_index.html'
            save.clear()
    elif index == 3:
        if 'fork' in save:
            page_selection = 'goodResult.html'
            save.clear()
        else:
            page_selection = 'bad_index.html'
            save.clear()
    elif index == 4:
        if 'mouse' in save:
            page_selection = 'goodResult.html'
            save.clear()
        else:
            page_selection = 'bad_index.html'
            save.clear()
    elif index == 5:
        if 'spoon' in save:
            page_selection = 'goodResult.html'
            save.clear()
        else:
            page_selection = 'bad_index.html'
            save.clear()
    elif index == 6:
        if 'toothbrush' in save:
            page_selection = 'goodResult.html'
            save.clear()
        else:
            page_selection = 'bad_index.html'
            save.clear()
    elif index == 7:
        if 'cell phone' in save:
            page_selection = 'goodResult.html'
            save.clear()
        else:
            page_selection = 'bad_index.html'
            save.clear()

    print(page_selection)

    # return generate_save()
    return render_template(page_selection)

@app.route('/')
def Fp():
    """Video streaming home page."""

    return render_template('firstpage.html')

@app.route('/manual')
def manual():

    return render_template('manual.html')

@app.route('/3sec')
def T3sec():

    return render_template('3sec.html')

@app.route('/2sec')
def T2sec():

    return render_template('2sec.html')

@app.route('/1sec')
def T1sec():

    return render_template('1sec.html')


@app.route('/game', methods=['GET','POST'])
def index():
    """Video streaming page."""


    global index

    index = random.randint(0, 7)

    # index = 0

    return render_template('index.html', index=index)

# def generate_save():
#     print("I'm in generate_save")
#     return_save = save.copy()
#     return_string = ','.join(return_save)
#     #save.clear()
#     return return_string

def gen_frames():  # generate frame by frame from camera


    frame_id = 0
    run_time = time.time()

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        # 영상 좌우반전
        frame = cv2.flip(frame, 1)
        frame_id += 1

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                if label != "person":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # 실제 웹에선 안 쓸 예정
                    # cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                    # cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)

                    # Detection realtime result
                    # print(label, confidence)
                    #리스트에 라벨 저장

                    if label not in save :
                        save.append(label)

                    print(save)


        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
        # cv2.imshow("Image", frame)

        # run_time = time.time()
        # key = cv2.waitKey(1)
        # if run_time-starting_time > 10:
        #     camera.release()



        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result








if __name__ == '__main__':
    app.run(debug=True)