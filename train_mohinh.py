import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Đường dẫn đến dữ liệu đã thu thập
train_data_dir = 'dataset/raw'

# Tạo bộ sinh dữ liệu và chuẩn hóa hình ảnh với data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Xây dựng mô hình CNN với dropout để giảm overfitting
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=30  # Tăng số lượng epochs nếu cần
)

# Lưu mô hình sau khi huấn luyện
model.save("Mohinh_nhandien.keras")
model.save("Mohinh_nhandien.h5")

# Lưu labels (class_indices) vào file JSON
labels = train_generator.class_indices
labels_inverted = {v: k for k, v in labels.items()}

# Lưu nhãn vào file JSON
with open("nhan.json", "w") as f:
    json.dump(labels_inverted, f)

print("Đã lưu mô hình và nhãn vào 'Mohinh_nhandien.keras' và 'nhan.json'")
