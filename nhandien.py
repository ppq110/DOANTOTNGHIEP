import cv2
import numpy as np
import tensorflow as tf
import json
import os
from datetime import datetime

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

# Lấy camera từ laptop
cap = cv2.VideoCapture(0)
# Lấy camera từ cam pi

if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

previous_labels = []
no_faces_detected = True  # Biến để theo dõi trạng thái không có ai -> có ai
history_dir = "history"

# Tạo thư mục history nếu chưa tồn tại
if not os.path.exists(history_dir):
    os.makedirs(history_dir)
    print("Thư mục history đã được tạo.")

def save_image(frame, label):
    # Lấy thời gian hiện tại
    current_time = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")  # Đổi định dạng để tránh dấu hai chấm
    # Tạo tên file ảnh với tên người và thời gian
    if label == "Chưa xác định":
        file_name = f"nguoi_la_{current_time}.jpg"
    else:
        file_name = f"{label}_{current_time}.jpg"

    file_path = os.path.join(history_dir, file_name)
    try:
        # Lưu ảnh
        cv2.imwrite(file_path, frame)
        print(f"Ảnh đã được lưu: {file_path}")
    except Exception as e:
        print(f"Lỗi khi lưu ảnh: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể lấy khung hình từ webcam.")
        break

    # Chuyển đổi khung hình sang màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_labels = []

    for (x, y, w, h) in faces:
        # Cắt và xử lý ảnh khuôn mặt
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

        confidence_text = f"{confidence * 100:.2f}%"
        current_labels.append(label)

        # Vẽ hình chữ nhật và ghi nhãn lên khung hình
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Chụp ảnh khi có sự thay đổi trạng thái từ không có ai -> có người hoặc từ người này -> người khác
    if len(faces) > 0 and no_faces_detected:
        print("Chụp ảnh: có người vào")
        save_image(frame.copy(), current_labels[0])  # Sử dụng frame.copy() để không làm thay đổi frame gốc
        no_faces_detected = False  # Đặt lại trạng thái là đã có người

    elif len(faces) == 0:
        print("Không có ai trong khung hình")
        no_faces_detected = True  # Đặt lại trạng thái khi không còn ai trong khung hình

    # Chụp ảnh khi có sự thay đổi từ người này sang người khác
    if current_labels != previous_labels and len(current_labels) > 0:
        print("Chụp ảnh: có sự thay đổi người")
        save_image(frame.copy(), current_labels[0])  # Sử dụng frame.copy() để không làm thay đổi frame gốc

    # Hiển thị khung hình video
    cv2.imshow("Frame", frame)

    previous_labels = current_labels

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
