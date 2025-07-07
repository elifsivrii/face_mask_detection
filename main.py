import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. VERİYİ HAZIRLA

# Görsellerin bulunduğu ana klasör
data_dir = "data"

# Görsel boyutu MobileNetV2 ile uyumlu olmalı: 224x224
img_size = (224, 224)

# Her adımda kaç görsel işleyeceğimizi belirliyoruz (mini batch boyutu)
batch_size = 32

# Görselleri otomatik yükleyen, normalize eden ve doğrulama setini ayıran araç
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Eğitim verisini hazırlıyoruz (verinin %80'i)
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',       # İkili sınıflandırma: maskeli / maskesiz
    subset='training',
    shuffle=True,
    seed=42
)

# Doğrulama verisini hazırlıyoruz (verinin %20'si)
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

#  2. MODELİ OLUŞTUR

# MobileNetV2 modelini daha önce ImageNet üzerinde eğitilmiş şekilde alıyoruz
# include_top=False: son sınıflandırma katmanlarını çıkarıyoruz (kendi sınıflarımızı koyacağız)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Önceden öğrenilmiş ağırlıkları değiştirmiyoruz

# Yeni modelimizi oluşturuyoruz
model = Sequential([
    base_model,                      # Önceden eğitilmiş model
    GlobalAveragePooling2D(),       # Özellikleri sıkıştırıp tek vektör yapar
    Dropout(0.4),                   # Aşırı öğrenmeyi azaltmak için rastgele nöron kapatma
    Dense(128, activation='relu'),  # Tam bağlantılı ara katman
    Dropout(0.2),                   # Yine dropout
    Dense(1, activation='sigmoid')  # Çıkış katmanı (0-1 arası değer: maske var mı yok mu)
])

# Modeli derliyoruz
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3. MODELİ EĞİT 

# Erken durdurma: doğrulama hatası 3 kere üst üste iyileşmezse dur
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Modeli eğitiyoruz
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[early_stop]
)

#  4. MODELİ KAYDET 

# Eğitilen modeli dosyaya kaydediyoruz
model.save("model.keras")
print(" Model başarıyla 'model.keras' olarak kaydedildi.")

# 5. EĞİTİM GRAFİKLERİNİ OLUŞTUR 

# Loss ve Accuracy grafiklerini çiziyoruz
plt.figure(figsize=(12,5))

# Loss (kayıp) grafiği
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

# Accuracy (doğruluk) grafiği
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()
print(" training_curves.png oluşturuldu.")

# 6. CONFUSION MATRIX & METRİKLER 

# Doğrulama verisinden tahminler alıyoruz
val_gen.reset()
pred_probs = model.predict(val_gen)
pred_labels = (pred_probs > 0.5).astype(int)  # 0.5 üzeriyse sınıf = 1
true_labels = val_gen.classes

# Confusion matrix çizimi
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["With Mask", "Without Mask"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Precision, recall, f1-score gibi metrikleri txt dosyasına kaydediyoruz
report = classification_report(true_labels, pred_labels, target_names=["With Mask", "Without Mask"])
with open("classification_report.txt", "w") as f:
    f.write(report)
print(" confusion_matrix.png ve classification_report.txt oluşturuldu.")

# 7. ÖRNEK GÖRSELLER ÜZERİNDE TAHMİN

# Validation görsellerini belleğe alıyoruz
val_images = []
val_labels = []

for i in range(len(val_gen)):
    imgs, labels = val_gen[i]
    for img, label in zip(imgs, labels):
        val_images.append(img)
        val_labels.append(int(label))

# Tüm görseller için tahmin yapıyoruz
val_preds = (model.predict(np.array(val_images)) > 0.5).astype(int)

# 2 maskeli + 3 maskesiz örnek seçiyoruz
mask_indices = [i for i, label in enumerate(val_labels) if label == 0]
no_mask_indices = [i for i, label in enumerate(val_labels) if label == 1]

chosen_mask = random.sample(mask_indices, 2)
chosen_no_mask = random.sample(no_mask_indices, 3)
chosen = chosen_mask + chosen_no_mask
random.shuffle(chosen)

# Görselleri çiziyoruz
plt.figure(figsize=(15,5))
for i, idx in enumerate(chosen):
    img = val_images[idx]
    true = "With Mask" if val_labels[idx] == 0 else "Without Mask"
    pred = "With Mask" if val_preds[idx] == 0 else "Without Mask"

    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True: {true}\nPred: {pred}")

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.close()
print("sample_predictions.png oluşturuldu.")

# TAMAMLANDI
print("\n Proje başarıyla tamamlandı. Oluşan dosyalar:\n- model.keras\n- training_curves.png\n- confusion_matrix.png\n- classification_report.txt\n- sample_predictions.png")
