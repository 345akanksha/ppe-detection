from flask import Flask, render_template, request, Response, redirect, url_for
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO(r"D:/ppe_detection/runs\detect/train6/weights/best.pt")

# Global variable for the uploaded file path
uploaded_file_path = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global uploaded_file_path

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '':
                uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(uploaded_file_path)
                return redirect(url_for('index'))

    return render_template('index.html', filename=uploaded_file_path)

# ✅ Webcam stream function
def generate_webcam():
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Couldn't access webcam")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ✅ Video stream function (uploaded file)
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/webcam_feed')
def webcam_feed():
    """ Webcam endpoint """
    return Response(generate_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    global uploaded_file_path
    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        return "No video uploaded", 400

    return Response(generate_frames(uploaded_file_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
