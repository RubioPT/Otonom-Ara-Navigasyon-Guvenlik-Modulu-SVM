# -*- coding: utf-8 -*-
"""
===============================================================================
  Otonom Arac Navigasyon Sistemi - Maksimum Marjli (SVM) Guvenlik Modulu
===============================================================================

  Bu modul, otonom bir aracin guzergahi uzerindeki engelleri iki sinifa ayirarak
  aralarinda en genis "guvenlik koridoru" olusturmayi hedefler.

  Kullanilan algoritma: Hard-Margin SVM (Destek Vektor Makinesi)
  Optimizasyon yontemi: Stochastic Gradient Descent (SGD)
  Hazir kutuphane kullanilmamistir (scikit-learn vb. yok).

  Moduller:
    - coordinate.py   : Coordinate veri yapisi
    - data_generator.py: DataGenerator - dogrusal ayrilabilen veri uretici
    - svm_model.py     : SVMModel - SVM egitim ve tahmin motoru
    - visualizer.py    : Visualizer - Matplotlib gorsellestirici

  Yazar: Otonom Navigasyon Takimi
  Tarih: 2026-04-22
===============================================================================
"""

import sys
import io

# Windows konsolunda Turkce karakter sorununu onlemek icin
# stdout'u UTF-8 olarak yeniden yapilandir
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from coordinate import Coordinate
from data_generator import DataGenerator
from svm_model import SVMModel
from visualizer import Visualizer


def main():
    """
    Ana calistirma fonksiyonu.

    Is akisi:
      1. Veri uretimi (DataGenerator - context manager ile)
      2. SVM modeli egitimi
      3. Model degerlendirmesi
      4. Gorsellestirme (Visualizer - context manager ile)

    Zaman Karmasikligi: O(E * N * d + G^2)
        E = epoch sayisi, N = veri sayisi, d = boyut, G = izgara cozunurlugu
    """

    print("=" * 70)
    print("  OTONOM ARAC NAVIGASYON SISTEMI")
    print("  Maksimum Marjli (SVM) Guvenlik Modulu")
    print("=" * 70)

    # -- 1. VERI URETIMI -------------------------------------------------------
    print("\n[1/4] Egitim verisi uretiliyor...")

    # Context manager ile bellek yonetimi
    with DataGenerator(seed=42) as generator:
        training_data = generator.generate(
            n_per_class=60,
            class1_center=(-2.5, -2.0),
            class2_center=(2.5, 2.0),
            spread=1.0,
        )

        # Veri kopyasi olustur (context manager cikisinda orijinal silinecek)
        data = list(training_data)

    print(f"  [OK] Toplam {len(data)} koordinat noktasi uretildi.")
    print(f"    - Sinif +1 (Guvenli): {sum(1 for p in data if p.label == 1)} adet")
    print(f"    - Sinif -1 (Engel):   {sum(1 for p in data if p.label == -1)} adet")

    # -- 2. MODEL EGITIMI ------------------------------------------------------
    print("\n[2/4] SVM modeli egitiliyor...")
    print("-" * 50)

    model = SVMModel(
        learning_rate=0.001,
        lambda_param=0.01,
        n_epochs=1000,
    )
    model.fit(data)

    print("-" * 50)

    # -- 3. MODEL DEGERLENDIRMESI -----------------------------------------------
    print("\n[3/4] Model degerlendiriliyor...")

    # Egitim seti dogrulugu
    results = model.evaluate(data)
    print(f"  Dogruluk: %{results['accuracy'] * 100:.1f}")
    print(f"  Dogru tahmin: {results['correct']}/{results['total']}")

    # Destek vektorleri
    support_vectors = model.get_support_vectors(data)
    print(f"  Destek vektoru sayisi: {len(support_vectors)}")

    # Ornek tahminler
    print("\n  Ornek tahminler:")
    test_points = [
        Coordinate(-3.0, -3.0),
        Coordinate(0.0, 0.0),
        Coordinate(3.0, 3.0),
        Coordinate(-1.0, 2.0),
        Coordinate(1.5, -1.5),
    ]
    for pt in test_points:
        prediction = model.predict(pt)
        decision = model.decision_function(pt.x, pt.y)
        label_str = "Guvenli [+]" if prediction == 1 else "Engel [-]"
        print(f"    ({pt.x:>5.1f}, {pt.y:>5.1f}) -> {label_str}  "
              f"(karar degeri: {decision:>7.3f})")

    # -- 4. GORSELLESTIRME -----------------------------------------------------
    print(f"\n[4/4] Gorsellestirme hazirlaniyor...")

    # Context manager ile Visualizer bellek yonetimi
    with Visualizer() as viz:
        # Ana grafik: Karar siniri + guvenlik koridoru
        viz.plot_decision_boundary(
            model=model,
            data=data,
            title="SVM Guvenlik Koridoru - Otonom Arac Navigasyonu",
            show=True,
            save_path=None,  # Kaydetmek icin: "svm_result.png"
        )

        # Egitim kayip grafigi
        viz.plot_training_loss(
            model=model,
            title="Egitim Kayip Fonksiyonu (Hinge Loss + L2 Reg.)",
            show=True,
            save_path=None,
        )

    print("\n" + "=" * 70)
    print("  Guvenlik modulu basariyla calistirildi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
