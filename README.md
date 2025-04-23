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
])
```
---
## 🔧 Compile & Training
```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```
---
## 📊 Hasil Evaluasi

Model mencapai **akurasi keseluruhan: 97.57%** pada data uji, dengan performa per kelas sebagai berikut:

| Kelas       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Anjing      | 0.9649    | 0.9821 | 0.9735   |
| Ayam        | 0.9928    | 0.9857 | 0.9892   |
| Domba       | 0.9472    | 0.9607 | 0.9539   |
| Gajah       | 0.9755    | 0.9964 | 0.9859   |
| Kucing      | 0.9892    | 0.9786 | 0.9838   |
| Kuda        | 0.9852    | 0.9536 | 0.9691   |
| Kupu-kupu   | 0.9927    | 0.9750 | 0.9838   |
| Laba-laba   | 0.9821    | 0.9821 | 0.9821   |
| Sapi        | 0.9505    | 0.9607 | 0.9556   |
| Tupai       | 0.9786    | 0.9821 | 0.9804   |

- **Akurasi total**: 97.57%  
- **Rata-rata F1-score**: 97.57%
---
## 💡 Kesimpulan
Model berhasil mengklasifikasikan gambar hewan dari 10 kelas dengan performa tinggi. Dengan memanfaatkan transfer learning dari EfficientNetV2S serta augmentasi data, model mampu mencapai akurasi lebih dari 97% meskipun dataset memiliki kompleksitas tinggi.
---
## 🚀 Teknologi yang Digunakan
- Python
- TensorFlow / Keras
- EfficientNetV2S (pre-trained model)
- Google Colab / Jupyter Notebook
