"""
Visualizer Modülü
=================
Otonom Araç Navigasyon Sistemi için görselleştirme sınıfı.

Matplotlib kullanarak:
  - İki sınıfın koordinat noktalarını
  - SVM karar sınırını (ayırıcı hiperdüzlem)
  - Güvenlik koridorunu (marj bölgesi)
  - Destek vektörlerini
  - Eğitim kayıp grafiğini
çizer.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from coordinate import Coordinate
from svm_model import SVMModel


class Visualizer:
    """
    SVM sonuçlarını görselleştiren sınıf.

    Context manager protokolünü destekler; 'with' bloğu ile
    kullanıldığında tüm figure nesneleri otomatik olarak
    kapatılır ve bellek serbest bırakılır.

    Attributes:
        _figures (list): Açılan matplotlib figure nesnelerinin referansları.
    """

    def __init__(self):
        """
        Visualizer oluşturur.

        Zaman Karmaşıklığı: O(1)
        """
        self._figures: list = []

    def __enter__(self):
        """
        Context manager giriş noktası.

        Zaman Karmaşıklığı: O(1)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager çıkış noktası.
        Tüm açık figure nesnelerini kapatarak bellek serbest bırakır.

        Zaman Karmaşıklığı: O(F) — F = açık figure sayısı
        """
        self.close_all()
        return False

    def close_all(self):
        """
        Tüm açık figure nesnelerini kapatır.

        Zaman Karmaşıklığı: O(F)
        """
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()

    def plot_decision_boundary(
        self,
        model: SVMModel,
        data: list,
        title: str = "SVM Güvenlik Koridoru — Otonom Araç Navigasyonu",
        show: bool = True,
        save_path: str = None,
    ):
        """
        SVM karar sınırını, marj bölgesini ve veri noktalarını çizer.

        Görselleştirme şunları içerir:
          1. Sınıf +1 noktaları (yeşil daireler — güvenli bölge)
          2. Sınıf -1 noktaları (kırmızı kareler — engel bölgesi)
          3. Karar sınırı (w·x + b = 0 çizgisi)
          4. Pozitif marj sınırı (w·x + b = +1)
          5. Negatif marj sınırı (w·x + b = -1)
          6. Güvenlik koridoru (marj arasındaki taralı bölge)
          7. Destek vektörleri (sarı kenarlı büyük noktalar)

        Args:
            model (SVMModel): Eğitilmiş SVM modeli.
            data (list[Coordinate]): Veri noktaları.
            title (str): Grafik başlığı.
            show (bool): Grafiği gösterip göstermeme.
            save_path (str, optional): Kaydedilecek dosya yolu.

        Zaman Karmaşıklığı: O(N + G)
            N = veri noktası sayısı, G = ızgara çözünürlüğü (renk haritası)
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        self._figures.append(fig)

        # ── Stil ayarları ──────────────────────────────────────────
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', which='both')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['right'].set_color('#30363d')

        # ── Veri noktalarını ayır ──────────────────────────────────
        # Zaman Karmaşıklığı: O(N)
        class_pos = [p for p in data if p.label == 1]
        class_neg = [p for p in data if p.label == -1]

        x_pos = [p.x for p in class_pos]
        y_pos = [p.y for p in class_pos]
        x_neg = [p.x for p in class_neg]
        y_neg = [p.y for p in class_neg]

        # ── Karar bölgesi arka plan renk haritası ─────────────────
        # Zaman Karmaşıklığı: O(G²) — G = ızgara çözünürlüğü
        all_x = [p.x for p in data]
        all_y = [p.y for p in data]
        x_min, x_max = min(all_x) - 1.5, max(all_x) + 1.5
        y_min, y_max = min(all_y) - 1.5, max(all_y) + 1.5

        xx = np.linspace(x_min, x_max, 300)
        yy = np.linspace(y_min, y_max, 300)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = np.zeros_like(XX)

        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                ZZ[i, j] = model.decision_function(XX[i, j], YY[i, j])

        # Karar bölgesi renk haritası
        ax.contourf(
            XX, YY, ZZ,
            levels=np.linspace(ZZ.min(), ZZ.max(), 50),
            cmap='RdYlGn',
            alpha=0.15,
        )

        # ── Marj sınırları ve karar çizgisi ───────────────────────
        # w1*x + w2*y + b = 0  →  y = -(w1*x + b) / w2
        # Zaman Karmaşıklığı: O(G)
        w1, w2 = model.w
        b = model.b

        if abs(w2) > 1e-10:
            x_line = np.linspace(x_min, x_max, 500)
            y_decision = -(w1 * x_line + b) / w2
            y_margin_pos = -(w1 * x_line + b - 1) / w2
            y_margin_neg = -(w1 * x_line + b + 1) / w2

            # Güvenlik koridoru (taralı bölge)
            ax.fill_between(
                x_line, y_margin_neg, y_margin_pos,
                alpha=0.12, color='#58a6ff',
                label=f'Güvenlik Koridoru (marj = {model.get_margin():.3f})',
            )

            # Karar sınırı
            ax.plot(
                x_line, y_decision,
                color='#f0f6fc', linewidth=2.5, linestyle='-',
                label='Karar Sınırı (w·x + b = 0)',
                zorder=5,
            )

            # Pozitif marj sınırı
            ax.plot(
                x_line, y_margin_pos,
                color='#58a6ff', linewidth=1.5, linestyle='--',
                label='Marj Sınırı (w·x + b = +1)',
                zorder=5,
            )

            # Negatif marj sınırı
            ax.plot(
                x_line, y_margin_neg,
                color='#58a6ff', linewidth=1.5, linestyle='--',
                label='Marj Sınırı (w·x + b = −1)',
                zorder=5,
            )

            # Y ekseni sınırları — karar çizgisi görünür kalsın
            ax.set_ylim(y_min, y_max)

        elif abs(w1) > 1e-10:
            # Dikey karar çizgisi durumu (w2 ≈ 0)
            x_decision = -b / w1
            x_margin_pos = -(b - 1) / w1
            x_margin_neg = -(b + 1) / w1

            ax.axvline(x=x_decision, color='#f0f6fc', linewidth=2.5,
                       label='Karar Sınırı')
            ax.axvline(x=x_margin_pos, color='#58a6ff', linewidth=1.5,
                       linestyle='--', label='Marj +1')
            ax.axvline(x=x_margin_neg, color='#58a6ff', linewidth=1.5,
                       linestyle='--', label='Marj −1')

            ax.axvspan(x_margin_neg, x_margin_pos, alpha=0.12, color='#58a6ff',
                       label=f'Güvenlik Koridoru (marj = {model.get_margin():.3f})')

        # ── Veri noktalarını çiz ──────────────────────────────────
        ax.scatter(
            x_pos, y_pos,
            c='#3fb950', marker='o', s=70, edgecolors='#238636',
            linewidths=1.2, alpha=0.85, zorder=10,
            label='Sınıf +1 (Güvenli Bölge)',
        )
        ax.scatter(
            x_neg, y_neg,
            c='#f85149', marker='s', s=70, edgecolors='#da3633',
            linewidths=1.2, alpha=0.85, zorder=10,
            label='Sınıf −1 (Engel Bölgesi)',
        )

        # ── Destek vektörlerini vurgula ───────────────────────────
        # Zaman Karmaşıklığı: O(N * d)
        support_vectors = model.get_support_vectors(data, tolerance=0.1)
        if support_vectors:
            sv_x = [sv.x for sv in support_vectors]
            sv_y = [sv.y for sv in support_vectors]
            ax.scatter(
                sv_x, sv_y,
                c='none', marker='o', s=200,
                edgecolors='#d2a8ff', linewidths=2.5,
                zorder=15, label=f'Destek Vektörleri ({len(support_vectors)} adet)',
            )

        # ── Başlık, etiketler ve lejant ───────────────────────────
        ax.set_title(
            title,
            fontsize=16, fontweight='bold', color='#f0f6fc',
            pad=20,
        )
        ax.set_xlabel('X Koordinatı', fontsize=12, color='#8b949e', labelpad=10)
        ax.set_ylabel('Y Koordinatı', fontsize=12, color='#8b949e', labelpad=10)

        legend = ax.legend(
            loc='upper left', fontsize=9,
            facecolor='#21262d', edgecolor='#30363d',
            labelcolor='#c9d1d9',
        )
        legend.get_frame().set_alpha(0.9)

        # Bilgi kutusu
        info_text = (
            f"w = [{w1:.4f}, {w2:.4f}]\n"
            f"b = {b:.4f}\n"
            f"||w|| = {(w1**2 + w2**2)**0.5:.4f}\n"
            f"Marj = {model.get_margin():.4f}"
        )
        ax.text(
            0.98, 0.02, info_text,
            transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            verticalalignment='bottom', horizontalalignment='right',
            color='#8b949e',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d',
                      edgecolor='#30363d', alpha=0.9),
        )

        ax.set_xlim(x_min, x_max)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  Grafik kaydedildi: {save_path}")

        if show:
            plt.show()

    def plot_training_loss(
        self,
        model: SVMModel,
        title: str = "Eğitim Kayıp Fonksiyonu",
        show: bool = True,
        save_path: str = None,
    ):
        """
        Epoch bazında eğitim kayıp grafiğini çizer.

        Args:
            model (SVMModel): Eğitilmiş SVM modeli.
            title (str): Grafik başlığı.
            show (bool): Grafiği gösterip göstermeme.
            save_path (str, optional): Kaydedilecek dosya yolu.

        Zaman Karmaşıklığı: O(E) — E = epoch sayısı
        """
        history = model.get_training_history()
        if not history:
            print("  Uyari: Egitim gecmisi bulunamadi.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        self._figures.append(fig)

        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', which='both')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        epochs = list(range(1, len(history) + 1))
        ax.plot(
            epochs, history,
            color='#58a6ff', linewidth=1.5, alpha=0.9,
        )
        ax.fill_between(epochs, history, alpha=0.1, color='#58a6ff')

        ax.set_title(title, fontsize=14, fontweight='bold', color='#f0f6fc', pad=15)
        ax.set_xlabel('Epoch', fontsize=11, color='#8b949e')
        ax.set_ylabel('Kayıp (Loss)', fontsize=11, color='#8b949e')

        ax.set_xlim(1, len(history))
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  Grafik kaydedildi: {save_path}")

        if show:
            plt.show()

    def __del__(self):
        """
        Nesne yok edilirken tüm figure'ları kapatır.

        Zaman Karmaşıklığı: O(F)
        """
        if hasattr(self, '_figures') and self._figures:
            self.close_all()
