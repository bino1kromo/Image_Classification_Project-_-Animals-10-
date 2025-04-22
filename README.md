# 🐾 Image Classification Project - Animals-10

Proyek ini merupakan implementasi klasifikasi gambar menggunakan deep learning berbasis CNN dengan arsitektur **EfficientNetV2S**. Dataset berisi sekitar **28.000 gambar** hewan berkualitas sedang yang terbagi dalam **10 kategori**.

---

## 📂 Dataset

Dataset ini terdiri dari 10 kelas hewan berikut:

- 🐶 **Anjing (Dog)**  
- 🐱 **Kucing (Cat)**  
- 🐴 **Kuda (Horse)**  
- 🕷️ **Laba-laba (Spider)**  
- 🦋 **Kupu-kupu (Butterfly)**  
- 🐔 **Ayam (Chicken)**  
- 🐑 **Domba (Sheep)**  
- 🐄 **Sapi (Cow)**  
- 🐿️ **Tupai (Squirrel)**  
- 🐘 **Gajah (Elephant)**  

📸 **Jumlah total gambar**: ~28.000

---

## 🧠 Model Architecture

Model dikembangkan menggunakan **EfficientNetV2S** (pre-trained dari ImageNet), kemudian ditambahkan beberapa layer untuk klasifikasi:

```python
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import layers, models

base_model = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])```

## 🔧 Compile & Training
python
Salin
Edit
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
