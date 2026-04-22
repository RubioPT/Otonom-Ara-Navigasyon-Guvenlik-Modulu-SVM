"""
SVM Model Modülü
================
Otonom Araç Navigasyon Sistemi için Destek Vektör Makinesi (SVM) implementasyonu.

Hiçbir hazır ML kütüphanesi kullanmadan, Gradyan İnişi (Gradient Descent) ile
||w||² değerini minimize ederek iki sınıf arasındaki marjı (güvenlik koridoru)
maksimize eden bir Hard-Margin SVM çözücü.

Matematiksel temel:
    Karar fonksiyonu: f(x) = w · x + b
    Amaç: min (1/2)||w||²
    Kısıt: y_i(w · x_i + b) >= 1,  ∀i

    Hinge Loss + Regularization ile yaklaşım:
        L = λ||w||² + (1/N) Σ max(0, 1 - y_i(w · x_i + b))
"""

import math
from coordinate import Coordinate


class SVMModel:
    """
    Hard-Margin SVM sınıflandırıcı.

    Gradyan İnişi (Stochastic Gradient Descent - SGD) kullanarak
    ağırlık vektörü w = [w1, w2] ve bias b değerini öğrenir.

    Attributes:
        w (list[float]): Ağırlık vektörü [w1, w2].
        b (float): Bias (yanlılık) terimi.
        learning_rate (float): Öğrenme oranı (η).
        lambda_param (float): Regularizasyon parametresi (λ).
        n_epochs (int): Eğitim epoch sayısı.
        _training_history (list): Epoch bazlı kayıp değerlerinin kaydı.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lambda_param: float = 0.01,
        n_epochs: int = 1000,
    ):
        """
        SVM modelini başlatır.

        Args:
            learning_rate (float): Gradyan inişi adım büyüklüğü.
            lambda_param (float): Regularizasyon katsayısı.
                                   Küçük değer → geniş marj, büyük değer → sıkı sınır.
            n_epochs (int): Eğitim iterasyon sayısı.

        Zaman Karmaşıklığı: O(1)
        """
        self.w: list = [0.0, 0.0]  # 2D için ağırlık vektörü
        self.b: float = 0.0
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self._training_history: list = []

    def _dot(self, a: list, b: list) -> float:
        """
        İki vektörün iç çarpımını (dot product) hesaplar.

        Args:
            a (list[float]): İlk vektör.
            b (list[float]): İkinci vektör.

        Returns:
            float: İç çarpım sonucu.

        Zaman Karmaşıklığı: O(d) — d = vektör boyutu (burada d=2)
        """
        result = 0.0
        for i in range(len(a)):
            result += a[i] * b[i]
        return result

    def _norm(self, v: list) -> float:
        """
        Vektörün L2 normunu (Öklid uzunluğu) hesaplar.

        Args:
            v (list[float]): Vektör.

        Returns:
            float: ||v||₂ değeri.

        Zaman Karmaşıklığı: O(d)
        """
        return math.sqrt(sum(vi * vi for vi in v))

    def fit(self, data: list) -> None:
        """
        SVM modelini verilen eğitim verisi üzerinde eğitir.

        Stochastic Gradient Descent (SGD) kullanarak Hinge Loss + L2 Regularization
        optimize eder:

            L = λ||w||² + (1/N) Σ max(0, 1 - y_i(w · x_i + b))

        Gradyan güncellemeleri:
            Eğer y_i(w · x_i + b) >= 1 (doğru sınıflandırma, marj dışı):
                w ← w - η(2λw)
                b ← b  (değişmez)

            Eğer y_i(w · x_i + b) < 1 (yanlış veya marj içi):
                w ← w - η(2λw - y_i · x_i)
                b ← b + η · y_i

        Args:
            data (list[Coordinate]): Eğitim verileri.

        Zaman Karmaşıklığı: O(E * N * d)
            E = epoch sayısı, N = veri sayısı, d = boyut (2)
        Alan Karmaşıklığı: O(E) — eğitim geçmişi için
        """
        n = len(data)
        if n == 0:
            raise ValueError("Eğitim verisi boş olamaz.")

        # Ağırlıkları sıfırla
        self.w = [0.0, 0.0]
        self.b = 0.0
        self._training_history = []

        for epoch in range(self.n_epochs):
            total_loss = 0.0

            for point in data:
                x_i = point.to_list()
                y_i = point.label

                # Karar değeri: w · x + b
                decision = self._dot(self.w, x_i) + self.b
                condition = y_i * decision

                if condition >= 1:
                    # Doğru sınıflandırma ve marj dışında
                    # Sadece regularizasyon gradyanı
                    # Zaman Karmaşıklığı (bu dal): O(d)
                    for j in range(len(self.w)):
                        self.w[j] -= self.learning_rate * (2 * self.lambda_param * self.w[j])
                else:
                    # Marj ihlali veya yanlış sınıflandırma
                    # Hinge loss + regularizasyon gradyanı
                    # Zaman Karmaşıklığı (bu dal): O(d)
                    for j in range(len(self.w)):
                        self.w[j] -= self.learning_rate * (
                            2 * self.lambda_param * self.w[j] - y_i * x_i[j]
                        )
                    self.b += self.learning_rate * y_i

                    # Hinge loss katkısı
                    total_loss += 1 - condition

            # Regularizasyon terimi
            reg_loss = self.lambda_param * self._dot(self.w, self.w)
            total_loss = reg_loss + total_loss / n

            self._training_history.append(total_loss)

            # Her 200 epoch'ta ilerleme bilgisi
            if (epoch + 1) % 200 == 0:
                margin = self.get_margin()
                print(
                    f"  Epoch {epoch + 1:>5}/{self.n_epochs} | "
                    f"Loss: {total_loss:.6f} | "
                    f"Marj: {margin:.4f} | "
                    f"||w||: {self._norm(self.w):.4f}"
                )

        print(f"\n  Egitim tamamlandi.")
        print(f"  Son agirliklar: w = [{self.w[0]:.4f}, {self.w[1]:.4f}], b = {self.b:.4f}")
        print(f"  Guvenlik koridoru (marj): {self.get_margin():.4f}")

    def predict(self, point: Coordinate) -> int:
        """
        Verilen koordinat için sınıf tahmini yapar.

        Karar fonksiyonu: sign(w · x + b)

        Args:
            point (Coordinate): Tahmin yapılacak nokta.

        Returns:
            int: Tahmin edilen sınıf (+1 veya -1).

        Zaman Karmaşıklığı: O(d) — d = boyut
        """
        x = point.to_list()
        decision = self._dot(self.w, x) + self.b
        return 1 if decision >= 0 else -1

    def decision_function(self, x: float, y: float) -> float:
        """
        Verilen (x, y) noktası için ham karar değerini döndürür.

        Args:
            x (float): X koordinatı.
            y (float): Y koordinatı.

        Returns:
            float: w · [x, y] + b değeri.

        Zaman Karmaşıklığı: O(d)
        """
        return self.w[0] * x + self.w[1] * y + self.b

    def get_margin(self) -> float:
        """
        Mevcut modelin marj (güvenlik koridoru) genişliğini hesaplar.

        Marj = 2 / ||w||

        Returns:
            float: Marj genişliği. ||w|| ≈ 0 ise sonsuz döndürür.

        Zaman Karmaşıklığı: O(d)
        """
        w_norm = self._norm(self.w)
        if w_norm < 1e-10:
            return float('inf')
        return 2.0 / w_norm

    def get_training_history(self) -> list:
        """
        Eğitim süresince kaydedilen kayıp değerlerini döndürür.

        Returns:
            list[float]: Epoch bazlı toplam kayıp değerleri.

        Zaman Karmaşıklığı: O(1)
        """
        return self._training_history

    def evaluate(self, data: list) -> dict:
        """
        Modelin doğruluğunu verilen veri seti üzerinde hesaplar.

        Args:
            data (list[Coordinate]): Test verileri.

        Returns:
            dict: {'accuracy': float, 'correct': int, 'total': int}

        Zaman Karmaşıklığı: O(N * d) — N = veri sayısı
        """
        if not data:
            return {'accuracy': 0.0, 'correct': 0, 'total': 0}

        correct = 0
        for point in data:
            if self.predict(point) == point.label:
                correct += 1

        total = len(data)
        return {
            'accuracy': correct / total,
            'correct': correct,
            'total': total,
        }

    def get_support_vectors(self, data: list, tolerance: float = 0.05) -> list:
        """
        Destek vektörlerini (marj sınırına en yakın noktalar) bulur.

        Bir nokta, |y_i(w · x_i + b) - 1| < tolerance ise
        destek vektörü olarak kabul edilir.

        Args:
            data (list[Coordinate]): Eğitim verileri.
            tolerance (float): Destek vektörü kabul toleransı.

        Returns:
            list[Coordinate]: Destek vektörleri listesi.

        Zaman Karmaşıklığı: O(N * d) — N = veri sayısı
        """
        support_vectors = []
        for point in data:
            x_i = point.to_list()
            y_i = point.label
            functional_margin = y_i * (self._dot(self.w, x_i) + self.b)
            if abs(functional_margin - 1.0) < tolerance:
                support_vectors.append(point)
        return support_vectors

    def __repr__(self) -> str:
        """
        Zaman Karmaşıklığı: O(1)
        """
        return (
            f"SVMModel(w=[{self.w[0]:.4f}, {self.w[1]:.4f}], "
            f"b={self.b:.4f}, margin={self.get_margin():.4f})"
        )

    def __del__(self):
        """
        Nesne yok edilirken eğitim geçmişini temizler.

        Zaman Karmaşıklığı: O(1)
        """
        if hasattr(self, '_training_history') and self._training_history is not None:
            self._training_history.clear()
