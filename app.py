from flask import Flask, render_template, Response
import cv2
import json
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import requests

app = Flask(__name__)

# Tải mô hình đã huấn luyện
try:
    model = tf.keras.models.load_model("Mohinh_nhandien.keras")
    print("Mô hình đã được tải thành công.")
except Exception as e:
    print(f"Không thể tải mô hình: {e}")

# Tải nhãn lớp từ file JSON
try:
    with open("nhan.json", "r") as f:
        labels = json.load(f)
    labels = {int(k): v for k, v in labels.items()}
    print("Nhãn lớp đã được tải thành công.")
except Exception as e:
    print(f"Không thể tải nhãn lớp: {e}")

# Khởi tạo bộ phát hiện khuôn mặt từ OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Địa chỉ URL API REST của Home Assistant
ha_url = "http://192.168.1.7:8123/api/states/sensor.nguoi_nhan_dien"
ha_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI0NTc5ZTg2OTVhMjk0YTNiOWVhNjk0MDQ5YzZjODU5MyIsImlhdCI6MTcyODU2NTk5NywiZXhwIjoyMDQzOTI1OTk3fQ.Mww7a6AIYEjPF18MXypO577PR2tJCyn1q_C-o7UICDA"

def generate_frames():
    cap = cv2.VideoCapture(0)
    no_faces_detected = True  # Biến để theo dõi trạng thái không có ai -> có ai
    previous_labels = []

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Chuyển đổi khung hình để nhận diện
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            current_labels = []

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (150, 150))
                face_img = np.expand_dims(face_img, axis=0) / 255.0

                # Dự đoán nhãn khuôn mặt
                predictions = model.predict(face_img)
                class_idx = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions)

                # Ngưỡng độ chính xác
                confidence_threshold = 0.5
                if confidence >= confidence_threshold:
                    label = labels.get(class_idx, "Chưa xác định")
                else:
                    label = "Chưa xác định"

                current_labels.append(label)

                # Vẽ hình chữ nhật và ghi nhãn lên khung hình
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Chụp ảnh khi có sự thay đổi trạng thái từ không có ai -> có người
            if len(faces) > 0 and no_faces_detected:
                print("Có người vào")
                send_to_home_assistant(current_labels[0])  # Gửi tên người lên Home Assistant
                no_faces_detected = False

            elif len(faces) == 0:
                print("Không có ai trong khung hình")
                no_faces_detected = True

            # Chụp ảnh khi có sự thay đổi từ người này sang người khác
            if current_labels != previous_labels and len(current_labels) > 0:
                print("Có sự thay đổi người")
                send_to_home_assistant(current_labels[0])  # Gửi tên người lên Home Assistant

            previous_labels = current_labels

            # Chuyển đổi khung hình thành định dạng JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def send_to_home_assistant(label):
    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }
    data = {
        "state": label,
        "attributes": {
            "friendly_name": "Người nhận diện",
            # Không gửi ảnh nữa
        }
    }
    response = requests.post(ha_url, headers=headers, json=data)
    if response.status_code == 200:
        print(f"Gửi trạng thái thành công lên Home Assistant: {label}")
    else:
        print(f"Error sending to Home Assistant: {response.status_code}, {response.text}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
