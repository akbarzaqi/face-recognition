import cv2
import numpy as np
from PIL import Image
import os


path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Memuat detektor wajah
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print(f"[ERROR] file tidak adaaa {cascade_path}")
else:
    detector = cv2.CascadeClassifier(cascade_path)

# Fungsi untuk mendapatkan gambar dan label
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png') or f.endswith('.jpg')]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L')  # Mengubah gambar menjadi grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")
    return faceSamples, ids



faces, ids = getImagesAndLabels(path)

print(f"[DEBUG] Jumlah wajah yang ditemukan: {len(faces)}")
print(f"[DEBUG] Jumlah id yang ditemukan: {len(ids)}")

if len(faces) == 0:
    print("[ERROR] Wajah tidak terdeteksi")
else:
    try:
       
        recognizer.train(faces, np.array(ids))

       
        if not os.path.exists('trainer'):
            os.makedirs('trainer')
        recognizer.write('trainer/trainer.yml')

      
        print("\n[INFO] {0} train done bang".format(len(np.unique(ids))))
    except cv2.error as e:
        print(f"[ERROR] aduhh error: {e}")
