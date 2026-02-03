from flask import Flask, Response, request
import cv2
import os
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)

# ================== HTML ==================

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Attendance</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: #f4f6f9;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
            padding: 30px;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        a.btn {
            margin: 10px;
            font-size: 16px;
            border-radius: 6px;
            transition: 0.3s;
        }
        a.btn:hover {
            transform: scale(1.05);
        }
        .video-frame {
            border: 3px solid #333;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Face Attendance System</h1>
    <a href="/register" class="btn btn-primary">Register New Person</a>
    <a href="/train" class="btn btn-success">Train Model</a>
    <br><br>
    <img src="/video" width="800" class="video-frame">
</body>
</html>
"""

REGISTER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Register</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: #e9ecef;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
            padding: 30px;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        form {
            background: #fff;
            padding: 20px;
            border-radius: 10px;
            display: inline-block;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
        }
        input {
            margin: 10px;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
            width: 250px;
        }
        button {
            background: #007bff;
            color: #fff;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 16px;
            transition: 0.3s;
        }
        button:hover {
            background: #0056b3;
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <h1>Register Face</h1>
    <form method="POST">
        <label><b>Name:</b></label>
        <input name="name" required>
        <br>
        <button type="submit">Start Capture</button>
    </form>
</body>
</html>
"""

# ================== FACE CONFIG ==================

FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 55

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=16,
    grid_x=8,
    grid_y=8
)

if os.path.exists("trainer/trainer.yml"):
    recognizer.read("trainer/trainer.yml")

marked_today = set()

# ================== HELPERS ==================

def preprocess_face(gray, x, y, w, h):
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.equalizeHist(face)
    return face

def mark_attendance(name):
    if name in marked_today:
        return

    os.makedirs("attendance", exist_ok=True)
    file = f"attendance/{datetime.now().date()}.csv"
    time = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(file):
        pd.DataFrame(columns=["Name", "Time"]).to_csv(file, index=False)

    df = pd.read_csv(file)
    df.loc[len(df)] = [name, time]
    df.to_csv(file, index=False)
    marked_today.add(name)

# ================== VIDEO ==================

def generate_frames():
    cam = cv2.VideoCapture(0)
    names = os.listdir("dataset") if os.path.exists("dataset") else []

    while True:
        ok, img = cam.read()
        if not ok:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(100,100))

        for (x,y,w,h) in faces:
            name = "Unknown"

            if os.path.exists("trainer/trainer.yml") and names:
                face = preprocess_face(gray, x, y, w, h)
                id, conf = recognizer.predict(face)

                if conf < CONFIDENCE_THRESHOLD and id < len(names):
                    name = names[id]
                    mark_attendance(name)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,f"{name}",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        ret, buffer = cv2.imencode(".jpg", img)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")

# ================== TRAINING ==================

def save_faces(name):
    path = f"dataset/{name}"
    os.makedirs(path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    count = 0

    while count < 60:
        ret, img = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(100,100))

        for (x,y,w,h) in faces:
            face = preprocess_face(gray, x, y, w, h)
            count += 1
            cv2.imwrite(f"{path}/{count}.jpg", face)

    cam.release()

def train_model():
    faces, ids = [], []
    current_id = 0

    for person in os.listdir("dataset"):
        for img_name in os.listdir(f"dataset/{person}"):
            img = cv2.imread(f"dataset/{person}/{img_name}", cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            ids.append(current_id)
        current_id += 1

    os.makedirs("trainer", exist_ok=True)
    recognizer.train(faces, np.array(ids))
    recognizer.save("trainer/trainer.yml")

# ================== ROUTES ==================

@app.route("/")
def index():
    return INDEX_HTML

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        save_faces(name)
        return f"<h2 style='color:green;'>Captured {name}</h2><a href='/train'>Train Model</a>"
    return REGISTER_HTML

@app.route("/train")
def train():
    train_model()
    return "<h2 style='color:blue;'>Training done</h2><a href='/'>Go Home</a>"

# ================== RUN ==================

if __name__ == "__main__":
    app.run(debug=True)