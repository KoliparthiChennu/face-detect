from keras.models import load_model
from flask import Flask, render_template, Response
import cv2
import numpy as np
model = load_model('Model/model_class_emotion_detector_V2.h5')

app=Flask(__name__)
def capture_by_frames(): 
    global camera
    camera = cv2.VideoCapture("0")
    detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
    int2emotions = {0:'angry',1:'fear',2:'happy',3:'neutral',4:'sad',5:'surprise',6:'disgust'}
    while True:
        success, frame = camera.read()  # read the camera frame
        faces=detector.detectMultiScale(frame,1.3,4)
         #Draw the rectangle around each face
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(172,42,251),2)
            face = frame[y:y+h,x:x+w]
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face,(48,48))
            face = face.reshape(1,48,48,1)
            cv2.putText(frame,text=int2emotions[np.argmax(model.predict(face))],org=(x,y-15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(106,40,243),thickness=2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/start',methods=['POST'])
def start():
    return render_template('index.html')
@app.route('/stop',methods=['POST'])
def stop():
    if camera.isOpened():
        camera.release()
    return render_template('stop.html')
@app.route('/video_capture')
def video_capture():
    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True,use_reloader=False, port=8000)
