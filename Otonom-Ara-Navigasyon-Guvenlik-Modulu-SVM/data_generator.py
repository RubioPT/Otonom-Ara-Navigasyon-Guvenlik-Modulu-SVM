"""
Data Generator Modülü
=====================
Otonom Araç Navigasyon Sistemi için doğrusal olarak ayrılabilen
iki farklı engel sınıfı üreten veri jeneratörü.

Üretilen veriler, gerçek bir otonom araç senaryosunda
"sol şerit engelleri" ve "sağ şerit engelleri" olarak düşünülebilir.
"""

import random
from coordinate import Coordinate


class DataGenerator:
    """
    Doğrusal olarak ayrılabilen (linearly separable) iki sınıf için
    rastgele koordinat verileri üreten sınıf.

    Bellek Yönetimi:
        Context manager (__enter__/__exit__) protokolünü destekler,
        böylece 'with' bloğu ile kullanıldığında üretilen veriler
        blok sonunda otomatik olarak temizlenir.
    """

    def __init__(self, seed: int = None):
        """
        DataGenerator oluşturur.

        Args:
            seed (int, optional): Rastgelelik tohumu. Tekrarlanabilir
                                   sonuçlar için kullanılır.

        Zaman Karmaşıklığı: O(1)
        """
        self._data: list = []
        self._seed = seed
        if seed is not None:
            random.seed(seed)

    def __enter__(self):
        """
        Context manager giriş noktası.

        Zaman Karmaşıklığı: O(1)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager çıkış noktası.
        Üretilen verileri temizleyerek bellek serbest bırakır.

        Zaman Karmaşıklığı: O(1) (GC referans sayısını düşürür)
        """
        self._data.clear()
        self._data = None
        return False

    def generate(
        self,
        n_per_class: int = 50,
        class1_center: tuple = (-2.0, -2.0),
        class2_center: tuple = (2.0, 2.0),
        spread: float = 0.8,
    ) -> list:
        """
        İki sınıf için doğrusal olarak ayrılabilen veri noktaları üretir.

        Noktalar, belirtilen merkez etrafında uniform dağılım ile saçılır.
        İki merkezin yeterince uzak olması doğrusal ayrılabilirliği garanti eder.

        Args:
            n_per_class (int): Her sınıf için üretilecek nokta sayısı.
            class1_center (tuple): Sınıf +1 merkez koordinatı (x, y).
            class2_center (tuple): Sınıf -1 merkez koordinatı (x, y).
            spread (float): Noktaların merkez etrafındaki saçılma yarıçapı.

        Returns:
            list[Coordinate]: Üretilen tüm koordinatların listesi.

        Zaman Karmaşıklığı: O(n) — n = 2 * n_per_class
        Alan Karmaşıklığı: O(n)
        """
        self._data = []

        # Sınıf +1: Güvenli bölge (sol alt küme)
        for _ in range(n_per_class):
            x = class1_center[0] + random.uniform(-spread, spread)
            y = class1_center[1] + random.uniform(-spread, spread)
            self._data.append(Coordinate(x, y, label=1))

        # Sınıf -1: Engel bölgesi (sağ üst küme)
        for _ in range(n_per_class):
            x = class2_center[0] + random.uniform(-spread, spread)
            y = class2_center[1] + random.uniform(-spread, spread)
            self._data.append(Coordinate(x, y, label=-1))

        return self._data

    def get_data(self) -> list:
        """
        Son üretilen veri setini döndürür.

        Returns:
            list[Coordinate]: Mevcut veri seti.

        Zaman Karmaşıklığı: O(1)
        """
        return self._data

    def __del__(self):
        """
        Nesne yok edilirken veri listesini temizler.

        Zaman Karmaşıklığı: O(1)
        """
        if hasattr(self, '_data') and self._data is not None:
            self._data.clear()
