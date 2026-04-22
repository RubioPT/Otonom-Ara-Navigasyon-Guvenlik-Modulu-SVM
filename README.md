# 🚀 Otonom Araç Navigasyon Güvenlik Modülü (SVM)

Bu proje, bir otonom aracın 2D düzlemde karşılaştığı engeller arasından en güvenli rotayı belirlemesini sağlayan, **Support Vector Machines (SVM)** tabanlı bir güvenlik modülüdür. Sistem, iki farklı engel sınıfını sadece ayırmakla kalmaz, her iki sınıfa olan uzaklığı maksimize ederek en geniş **"Güvenlik Koridoru"** oluşturur.

## 📌 Projenin Amacı
Otonom sistemlerde sensör gürültüsü ve ölçüm hataları kaçınılmazdır. Bu modül, engeller arasından geçilebilecek en uzak orta noktayı (Maximum Margin Hyperplane) bularak, aracın sensör hatalarına karşı maksimum toleransla hareket etmesini sağlar.

## 🛠️ Yazılım Mimarisi (OOP)
Proje, yazılım mühendisliği prensiplerine uygun olarak katmanlı ve modüler bir yapıda tasarlanmıştır:

* **`Coordinate`**: Veri noktalarını ve etiketlerini temsil eden hafif veri yapısı.
* **`SVMModel`**: Stokastik Gradyan İnişi (SGD) kullanarak optimizasyonu gerçekleştiren ana motor.
* **`DataGenerator`**: Test süreçleri için doğrusal ayrılabilir yapay engel verileri üreten modül.
* **`Visualizer`**: Karar sınırlarını, marj boşluklarını ve destek vektörlerini görselleştiren raporlama aracı.

## 🧠 Algoritma ve Analiz
* **Model:** Hard-Margin SVM (Sıfırdan implemente edilmiştir).
* **Optimizasyon:** Hinge Loss fonksiyonu üzerinden Gradyan İnişi.
* **Zaman Karmaşıklığı:**
    * **Eğitim:** $O(N^2)$
    * **Tahmin:** $O(1)$ (Gerçek zamanlı navigasyon için optimize edilmiştir).

## 💾 Bellek Yönetimi
* **`__slots__` Kullanımı:** Nesne başına bellek kullanımını minimize ederek RAM verimliliği artırılmıştır.
* **Deterministik Temizlik:** `__del__` metotları ile kullanılmayan kaynakların temizlenmesi sağlanmıştır.
* **Vektörel İşlemler:** Gereksiz liste kopyalamalarından kaçınılarak "sıfır bellek sızıntısı" hedeflenmiştir.
✍️ Hazırlayan
Hikmet Buğra Uzun - Kırklareli Üniversitesi, Yazılım Mühendisliği Bölümü 2. Sınıf Öğrencisi
