# 🚗 Güvenli Sürüş Takip Sistemi

Derin Öğrenme ve Yapay Zeka Tabanlı Sürücü Dikkat Dağınıklığı ve Uyku Tespiti Projesi

Bu proje, trafik kazalarının en büyük nedenlerinden biri olan sürücü yorgunluğu ve dikkat dağınıklığını önlemek amacıyla geliştirilmiş gerçek zamanlı bir bilgisayarlı görü ve yapay zeka sistemidir.

## 🌟 Özellikler

* **Uyku ve Yorgunluk Tespiti:** Göz En-Boy Oranı (EAR) ve Ağız En-Boy Oranı (MAR) metrikleri kullanılarak anlık mikro uyku ve esneme takibi.
* **Dikkat Dağınıklığı Analizi:** Sürücünün baş pozisyonu izlenerek yoldan uzun süreli göz ayırma durumlarının tespiti.
* **Telefon Kullanımı Tespiti:** YOLOv8 nesne tanıma modeli ile direksiyon başında telefon kullanımı ve kulakta telefonla konuşma analizi.
* **Gerçek Zamanlı Uyarı ve Raporlama:** Tehlike anında sesli alarm devreye girer. Kural ihlalleri anlık olarak fotoğraflanır ve her sürüş oturumu detaylı bir şekilde `.txt` formatında loglanır.
* **Kişiselleştirilebilir Otomatik Kalibrasyon:** Her sürücünün yüz yapısına göre sistemi otomatik ayarlayan kalibrasyon modülü.

## 🛠️ Kullanılan Teknolojiler

* **Python 3.x**
* **Streamlit:** Web tabanlı kullanıcı arayüzü
* **OpenCV:** Görüntü işleme ve kamera entegrasyonu
* **MediaPipe (Face Mesh):** Yüz ve göz işaret noktalarının (landmarks) tespiti
* **PyTorch:** Özel göz durumu (açık/kapalı) sınıflandırma CNN modeli (`eye_model.pt`)
* **Ultralytics YOLO:** Hızlı ve tutarlı nesne/telefon tespiti (`yolov8n.pt`)

## 📁 Proje Dosya Yapısı

```text
├── web_app.py               # Streamlit arayüzü ve ana kontrol döngüsü
├── requirements.txt         # Proje bağımlılıkları
├── eye_model.pt             # Göz sınıflandırması için eğitilmiş CNN modeli ağırlıkları
├── yolov8n.pt               # Nesne tespiti için YOLO modeli
├── face_landmarker.task     # MediaPipe yüz referans modeli
├── uyari_goruntuleri/       # Kural ihlalinde alınan otomatik ekran görüntülerinin kaydedildiği dizin
└── surus_kayitlari/         # Oturum sonu detaylı sürüş raporlarının (.txt) saklandığı dizin
