import os
import datetime


def delete_history_files(target_date):
    # Đường dẫn đến thư mục history
    history_dir = "history"

    # Duyệt qua tất cả các file trong thư mục history
    for filename in os.listdir(history_dir):
        file_path = os.path.join(history_dir, filename)

        # Kiểm tra nếu đây là file và có tên theo định dạng
        if os.path.isfile(file_path):
            # Phân tách thông tin từ tên file
            try:
                # Lấy thông tin ngày từ tên file
                date_part = filename.split("_")[-1].replace(".jpg", "")
                file_date = datetime.datetime.strptime(date_part, "%d-%m-%Y")

                # Kiểm tra xem file có tạo trước hoặc vào ngày mục tiêu không
                if file_date <= target_date:
                    try:
                        os.remove(file_path)
                        print(f"Đã xóa file: {filename}")
                    except Exception as e:
                        print(f"Lỗi khi xóa file {filename}: {e}")
            except ValueError:
                print(f"Không thể phân tích tên file: {filename}")


if __name__ == "__main__":
    # Nhập ngày theo định dạng DD/MM/YYYY
    date_input = input("Nhập dữ liệu ngày muốn xoá: ")

    # Chuyển đổi chuỗi nhập vào thành đối tượng ngày
    try:
        target_date = datetime.datetime.strptime(date_input, "%d/%m/%Y")
        # Xóa các file trong thư mục history
        delete_history_files(target_date)
    except ValueError:
        print("Định dạng ngày không hợp lệ. Vui lòng nhập theo định dạng DD/MM/YYYY.")
