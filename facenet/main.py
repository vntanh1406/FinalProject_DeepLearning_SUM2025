import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras_facenet import FaceNet
import pickle

# Load detector, embedder, label encoder, model
detector = MTCNN()
embedder = FaceNet()

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)

embedding_dim = 512
model = Sequential([
    Dense(128, activation='relu', input_shape=(embedding_dim,)),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('proxy_classifier_best.h5')

# Trích xuất khuôn mặt
def extract_face(image, required_size=(160,160)):
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    face_image = Image.fromarray(face)
    face_image = face_image.resize(required_size)
    return np.asarray(face_image)

# Lấy embedding
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    embedding = embedder.embeddings(np.expand_dims(face_pixels, axis=0))
    return embedding[0]

# Dự đoán
def predict(image):
    face = extract_face(image)
    if face is None:
        return None, None
    embedding = get_embedding(face)
    pred = model.predict(np.expand_dims(embedding, axis=0))
    class_idx = np.argmax(pred)
    class_name = label_encoder.inverse_transform([class_idx])[0]
    confidence = pred[0][class_idx]
    return class_name, confidence

# GUI
class FaceRecApp:
    def __init__(self, root):
        self.root = root
        root.title("Face Recognition - Cute Pink Theme")
        root.configure(bg='#FFC0CB')  # Màu hồng pastel

        self.font_title = ('Comic Sans MS', 18, 'bold')
        self.font_normal = ('Comic Sans MS', 14)

        self.label = tk.Label(root, text="Load an image to recognize face",
                              font=self.font_title, bg='#FFC0CB', fg='#C71585')
        self.label.pack(pady=(20,10))

        self.load_btn = tk.Button(root, text="🌸 Load Image 🌸", command=self.load_image,
                                  font=self.font_normal,
                                  bg='#FF69B4', fg='white', activebackground='#FF1493',
                                  relief='flat', padx=20, pady=10)
        self.load_btn.pack(pady=(0, 20))

        self.image_panel = tk.Label(root, bg='#FFC0CB')
        self.image_panel.pack()

        self.result_label = tk.Label(root, text="", font=self.font_normal,
                                     bg='#FFC0CB', fg='#8B008B')
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        image = Image.open(file_path).convert('RGB')

        max_size = 400
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        width, height = image.size
        left = max(0, (width - max_size) // 2)
        top = max(0, (height - max_size) // 2)
        right = left + max_size
        bottom = top + max_size

        if width < max_size or height < max_size:
            new_img = Image.new('RGB', (max_size, max_size), (255, 255, 255))
            paste_x = (max_size - width) // 2
            paste_y = (max_size - height) // 2
            new_img.paste(image, (paste_x, paste_y))
            image_to_show = new_img
        else:
            image_to_show = image.crop((left, top, right, bottom))

        self.photo = ImageTk.PhotoImage(image_to_show)
        self.image_panel.config(image=self.photo)

        class_name, confidence = predict(image)
        if class_name is None:
            self.result_label.config(text="No face detected!", fg='red')
        else:
            special_names = {
                'anh': "Zo Ann",
                'hieu': "Hiu Fam",
                'tien': "Zinh Tien",
                'long': "Anh Lonk singer"
            }
            display_name = special_names.get(class_name.lower(), class_name)
            self.result_label.config(
                text=f"Predicted: {display_name}\nConfidence: {confidence:.2f}", fg='green'
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecApp(root)
    root.geometry("500x650")
    root.mainloop()