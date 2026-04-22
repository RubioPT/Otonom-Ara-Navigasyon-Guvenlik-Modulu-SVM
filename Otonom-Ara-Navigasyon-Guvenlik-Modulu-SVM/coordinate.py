"""
Coordinate Modülü
=================
Otonom Araç Navigasyon Sistemi için koordinat veri yapısını tanımlar.
Her koordinat noktası (x, y) konumunu ve sınıf etiketini (+1 veya -1) tutar.
"""


class Coordinate:
    """
    2D düzlem üzerinde bir noktayı temsil eden veri yapısı.

    Attributes:
        x (float): X ekseni koordinatı.
        y (float): Y ekseni koordinatı.
        label (int): Sınıf etiketi (+1: güvenli bölge, -1: engel bölgesi).

    Bellek Yönetimi:
        __slots__ kullanılarak dinamik __dict__ oluşturulması engellenir,
        bu sayede her Coordinate nesnesi daha az bellek tüketir.
        Binlerce engel noktası oluşturulduğunda bu fark belirgin hale gelir.
    """

    __slots__ = ('x', 'y', 'label')

    def __init__(self, x: float, y: float, label: int = 0):
        """
        Coordinate nesnesi oluşturur.

        Args:
            x (float): X koordinatı.
            y (float): Y koordinatı.
            label (int): Sınıf etiketi (+1 veya -1). Varsayılan 0 (etiketsiz).

        Zaman Karmaşıklığı: O(1)
        Alan Karmaşıklığı: O(1)
        """
        self.x = x
        self.y = y
        self.label = label

    def to_list(self) -> list:
        """
        Koordinatı [x, y] listesi olarak döndürür.

        Returns:
            list: [x, y] formatında koordinat.

        Zaman Karmaşıklığı: O(1)
        """
        return [self.x, self.y]

    def __repr__(self) -> str:
        """
        Zaman Karmaşıklığı: O(1)
        """
        return f"Coordinate(x={self.x:.2f}, y={self.y:.2f}, label={self.label})"

    def __eq__(self, other) -> bool:
        """
        İki koordinatın eşitliğini kontrol eder.

        Zaman Karmaşıklığı: O(1)
        """
        if not isinstance(other, Coordinate):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.label == other.label

    def __hash__(self) -> int:
        """
        Koordinatı hashlenebilir yapar (set/dict kullanımı için).

        Zaman Karmaşıklığı: O(1)
        """
        return hash((self.x, self.y, self.label))

    def __del__(self):
        """
        Nesne yok edilirken çağrılır.
        __slots__ kullandığımız için zaten minimal bellek kullanıyoruz,
        ancak bellek temizliği farkındalığını göstermek adına tanımlanmıştır.

        Zaman Karmaşıklığı: O(1)
        """
        pass  # __slots__ sayesinde ekstra temizlik gerekmez
