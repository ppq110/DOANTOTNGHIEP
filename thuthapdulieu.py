import cv2
import os

# Tạo thư mục để lưu trữ dữ liệu nếu chưa tồn tại
DATASET_DIR = "dataset/raw"
os.makedirs(DATASET_DIR, exist_ok=True)

# Tạo thư mục mới nếu có trùng tên
def get_unique_name(name, base_dir):
    count = 1
    new_name = name
    while os.path.exists(os.path.join(base_dir, new_name)):
        new_name = f"{name}_{count}"
        count += 1
    return new_name

# Xoá thư mục
def delete_person_data(name, base_dir):
    person_dir = os.path.join(base_dir, name)
    if os.path.exists(person_dir):
        confirmation = input(f"Bạn có chắc chắn muốn xóa dữ liệu của '{name}'? (y/n): ").strip().lower()
        if confirmation == 'y':
            os.rmdir(person_dir)  # Xóa thư mục
            print(f"Đã xóa dữ liệu của '{name}'")
        else:
            print(f"Không xóa dữ liệu của '{name}'")
    else:
        print(f"Dữ liệu về '{name}' không tồn tại")

# Lấy camera từ laptop
cap = cv2.VideoCapture(0)
# Lấy camera từ cam pi

# Kiểm tra camera có mở được không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Lấy tên cần tạo thư mục
name = input("Nhập tên của bạn (hoặc nhập 'xoa' để xóa dữ liệu): ").strip()

# Nếu nhập xoá thì thực hiện việc xoá thư mục
if name.lower() == 'xoa':
    person_to_delete = input("Nhập tên của người muốn xóa: ").strip()
    delete_person_data(person_to_delete, DATASET_DIR)
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Tạo tên thư mục và lưu trong dataset
person_dir = os.path.join(DATASET_DIR, name)

# Kiểm tra xem tên đã tồn tại chưa và xử lý theo phản hồi của người dùng
if os.path.exists(person_dir):
    response = input(f"Tên '{name}' đã tồn tại. Bạn có muốn ghi đè dữ liệu không? (y/n): ").strip().lower()
    if response == 'y':
        print(f"Đã xóa dữ liệu của '{name}'")
    else:
        name = get_unique_name(name, DATASET_DIR)
        person_dir = os.path.join(DATASET_DIR, name)

# Tạo thư mục mới cho người dùng
os.makedirs(person_dir, exist_ok=True)

count = 0
print("Nhấn 'c' để thu thập hình ảnh, nhấn 'space' để dừng.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận frame từ camera. Thoát chương trình...")
        break

    # Hiển thị frame video trực tiếp từ camera
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Khi người dùng nhấn phím 'c'
        # Chuyển đổi ảnh thành thang xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện các khuôn mặt trong ảnh
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:  # Chỉ lưu khi có khuôn mặt
            for (x, y, w, h) in faces:
                # Cắt ảnh khuôn mặt từ frame gốc để lưu
                face_img = frame[y:y + h, x:x + w]
                img_path = os.path.join(person_dir, f"{count}.jpg")
                cv2.imwrite(img_path, face_img)
                print(f"Lưu hình ảnh: {img_path}")
                count += 1
        else:
            print("Không phát hiện khuôn mặt trong frame.")

    if key == ord(' '):  # Khi người dùng nhấn phím 'space' để ngừng
        break

# Giải phóng camera và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()