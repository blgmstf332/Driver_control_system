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

🚀 Kurulum ve Çalıştırma Yönergesi
Projeyi bilgisayarınızda yerel olarak çalıştırmak için aşağıdaki adımları sırasıyla uygulayınız:
1. Projeyi Bilgisayarınıza İndirin:
Terminali veya komut satırını açın ve projeyi klonlayın (veya GitHub üzerinden ZIP olarak indirip klasöre çıkartın):
git clone <github-depo-linkiniz-buraya-gelecek>
2. Proje Klasörüne Girin:
Terminal üzerinden projenin bulunduğu klasörün içine girin:
cd surucu
3. Gerekli Kütüphaneleri Kurun:
Sistemin çalışması için gerekli olan Python kütüphanelerini tek seferde kurmak için aşağıdaki komutu çalıştırın:
pip install -r requirements.txt
4. Uygulamayı Başlatın:
Tüm kurulumlar tamamlandıktan sonra, uygulamayı başlatmak için şu komutu çalıştırın:
streamlit run web_app.py
Not: Bu komutu çalıştırdıktan sonra tarayıcınızda otomatik olarak bir sekme açılacak ve sistem arayüzü karşınıza gelecektir.

## 📁 Proje Dosya Yapısı

```text
├── web_app.py               # Streamlit arayüzü ve ana kontrol döngüsü
├── requirements.txt         # Proje bağımlılıkları
├── eye_model.pt             # Göz sınıflandırması için eğitilmiş CNN modeli ağırlıkları
├── yolov8n.pt               # Nesne tespiti için YOLO modeli
├── face_landmarker.task     # MediaPipe yüz referans modeli
├── uyari_goruntuleri/       # Kural ihlalinde alınan otomatik ekran görüntülerinin kaydedildiği dizin
└── surus_kayitlari/         # Oturum sonu detaylı sürüş raporlarının (.txt) saklandığı dizin
