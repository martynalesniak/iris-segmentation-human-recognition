
# main.py – updated iris detection with ROI‑based thresholding
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTabWidget, QLabel, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from operations import (
    grayscale, binarization, erosion, dilatation, opening, closing,
    horizontal_projection, vertical_projection, polar_to_cartesian, close_and_fill, refine_circle_and_score,
    circularity_and_completeness, largest_connected_component
)

# ---------------- helper -------------------------------------------------

def detect_iris_from_pupil(gray_image, pupil_center, pupil_radius):
    """Wykrywa tęczówkę metodą gradientową, startując od krawędzi źrenicy."""
    h, w = gray_image.shape
    max_search_radius = int(pupil_radius * 3.2)  # Maksymalny promień poszukiwań
    
    # Przygotowanie tablicy na profile intensywności
    num_angles = 36  # Liczba kierunków próbkowania
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    iris_radius_estimates = []
    
    # Dla każdego kąta
    for angle in angles:
        # Współrzędne wektora kierunkowego
        dx, dy = np.cos(angle), np.sin(angle)
        
        # Tablica na profil intensywności dla tego kąta
        intensity_profile = []
        radius_values = []
        
        # Próbkuj punkty wzdłuż promienia
        for r in range(pupil_radius + 5, max_search_radius, 2):  # Zacznij poza źrenicą
            # Oblicz współrzędne punktu
            x = int(pupil_center[0] + dx * r)
            y = int(pupil_center[1] + dy * r)
            
            # Sprawdź czy punkt jest w granicach obrazu
            if 0 <= x < w and 0 <= y < h:
                intensity = gray_image[y, x]
                intensity_profile.append(intensity)
                radius_values.append(r)
        
        # Jeśli zebraliśmy wystarczająco próbek
        if len(intensity_profile) > 10:
            # Wygładź profil intensywności
            smoothed = np.convolve(intensity_profile, np.ones(5)/5, mode='valid')
            
            # Oblicz gradient (pierwszą pochodną) intensywności
            gradient = np.gradient(smoothed)
            
            # Znajdź indeks maksymalnej zmiany gradientu (to może wskazywać granicę tęczówki)
            # Szukamy tylko w sensownym zakresie (pomijamy początek i koniec profilu)
            search_range = len(gradient) // 3  # Pomiń pierwszy 1/3 profilu
            if search_range < len(gradient):
                # Znajdź pozycję największej zmiany intensywności
                max_grad_idx = search_range + np.argmax(np.abs(gradient[search_range:]))
                if max_grad_idx < len(radius_values):
                    # Dodaj estymację promienia tęczówki
                    iris_radius_estimates.append(radius_values[max_grad_idx])
    
    # Usuń skrajne wartości i oblicz średnią
    if iris_radius_estimates:
        # Usuń 20% skrajnych wartości
        sorted_estimates = sorted(iris_radius_estimates)
        num_to_remove = len(sorted_estimates) // 5
        filtered_estimates = sorted_estimates[num_to_remove:-num_to_remove] if num_to_remove > 0 else sorted_estimates
        
        # Średnia z pozostałych estymacji
        iris_radius = int(np.mean(filtered_estimates))
        return iris_radius
    else:
        # Jeśli nie znaleziono estymacji, zwróć domyślną wartość
        return int(pupil_radius * 2.5)
    
def binarization_ring(image, center, inner_r, outer_r, factor):
    """Mean‑based threshold inside an annulus (pure NumPy)."""
    gray = grayscale(image)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    ring = (dist2 >= inner_r ** 2) & (dist2 <= outer_r ** 2)
    P = np.mean(gray[ring])
    thr = P / factor
    out = np.zeros_like(gray, dtype=np.uint8)
    out[gray < thr] = 255
    return out


def cv_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        h, w = img.shape; q = QImage(img.data, w, h, w, QImage.Format_Indexed8)
    else:
        h, w, c = img.shape; q = QImage(img.data, w, h, c * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q)


class IrisUI(QMainWindow):
    TAB_TITLES = [
        "Step 0: Original", "Step 1: Pupil Binary", "Step 2: Pupil Circle",
        "Step 3: Iris Binary", "Step 4: Pupil + Iris Circles", "Step 5: Unwrapped Iris"
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris Segmentation Pipeline")
        scr = QApplication.primaryScreen().size()
        self.max_w, self.max_h = scr.width() // 2, scr.height() // 2
        self._reset_state()
        self._build_ui()

    # ------------- state -----------------------------------------------
    def _reset_state(self):
        self.orig = self.gray = None
        self.bin_pupil = self.bin_iris = None
        self.pcen = self.prad = None
        self.irad = None
        self.overlay_p = self.overlay_both = None
        self.unwrapped = None

    # ------------- UI ---------------------------------------------------
    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c); v = QVBoxLayout(c)
        btn = QPushButton("Load Image"); btn.clicked.connect(self.load_image); v.addWidget(btn)
        self.tabs = QTabWidget(); v.addWidget(self.tabs)
        self.labels = []
        for t in self.TAB_TITLES:
            lab = QLabel(alignment=Qt.AlignCenter)
            lab.setScaledContents(True)
            tab = QWidget(); QVBoxLayout(tab).addWidget(lab)
            self.tabs.addTab(tab, t); self.labels.append(lab)
        self.tabs.currentChanged.connect(self.update_tab)

    def load_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select eye image", "", "Images (*.png *.jpg *.jpeg)")
        if not p: return
        self.orig = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        self.gray = grayscale(self.orig)
        self._reset_state(); self.orig = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); self.gray = grayscale(self.orig)
        self._show(0, self.orig)
        self.tabs.setCurrentIndex(0)

    # ------------- tab dispatcher --------------------------------------
    def update_tab(self, idx):
        if self.orig is None: return
        if idx == 0: self._show(0, self.orig)
        elif idx == 1:
            if self.bin_pupil is None: self._compute_pupil()
            self._show(1, self.bin_pupil)
        elif idx == 2:
            if self.overlay_p is None: self._compute_pupil_overlay()
            self._show(2, self.overlay_p)
        elif idx == 3:
            if self.bin_iris is None: self._compute_iris()
            self._show(3, self.bin_iris)
        elif idx == 4:
            if self.overlay_both is None: self._compute_both_overlay()
            self._show(4, self.overlay_both)
        elif idx == 5:
            if self.unwrapped is None: self._compute_unwrap()
            self._show(5, self.unwrapped)

    # ------------- show scaled pixmap -----------------------------------
    def _show(self, slot, img):
        pix = cv_to_qpixmap(img).scaled(self.max_w, self.max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.labels[slot].setPixmap(pix)

    def remove_eyelids(self, image, iris_center, iris_radius):
        """Prosta metoda usuwania powiek przez wycięcie stałych segmentów
        górnej i dolnej części tęczówki.
        """
        # Stwórz kopię obrazu
        result = image.copy()
        h, w = image.shape[:2]
        
        # Utwórz pustą maskę
        eyelid_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Podziel tęczówkę na segmenty
        angle_step = 20  # W stopniach
        
        # Usuwamy górną część (od -60° do 60°)
        for angle in range(-60, 61, angle_step):
            rad = np.radians(angle)
            start_x = int(iris_center[0] + (iris_radius * 0.3) * np.cos(rad))
            start_y = int(iris_center[1] + (iris_radius * 0.3) * np.sin(rad))
            end_x = int(iris_center[0] + iris_radius * np.cos(rad))
            end_y = int(iris_center[1] + iris_radius * np.sin(rad))
            
            # Rysuj linie od środka tęczówki do jej krawędzi
            cv2.line(eyelid_mask, (start_x, start_y), (end_x, end_y), 255, 2)
        
        # Usuwamy dolną część (od 120° do 240°)
        for angle in range(120, 241, angle_step):
            rad = np.radians(angle)
            start_x = int(iris_center[0] + (iris_radius * 0.3) * np.cos(rad))
            start_y = int(iris_center[1] + (iris_radius * 0.3) * np.sin(rad))
            end_x = int(iris_center[0] + iris_radius * np.cos(rad))
            end_y = int(iris_center[1] + iris_radius * np.sin(rad))
            
            # Rysuj linie od środka tęczówki do jej krawędzi
            cv2.line(eyelid_mask, (start_x, start_y), (end_x, end_y), 255, 2)
        
        # Rozszerz maskę, aby utworzyć większe obszary do usunięcia
        kernel = np.ones((7, 7), np.uint8)
        eyelid_mask = cv2.dilate(eyelid_mask, kernel, iterations=3)
        
        # Zastosuj maskę do obrazu
        if len(image.shape) > 2:  # Obraz kolorowy
            # Rozszerz maskę do 3 kanałów
            eyelid_mask_color = cv2.merge([eyelid_mask, eyelid_mask, eyelid_mask])
            
            # Zastąp obszary powiek białym kolorem
            white_bg = np.ones_like(result) * 255
            result = np.where(eyelid_mask_color > 0, white_bg, result)
        else:  # Obraz w skali szarości
            # Zastąp obszary powiek białym kolorem
            result[eyelid_mask > 0] = 255
        
        return result, cv2.bitwise_not(eyelid_mask)
    # ------------- pupil processing -------------------------------------

    def _compute_pupil(self):
        best_score = 0
        best_mask = None
        factors = [i / 10 for i in range(29, 60, 3)]
        kernel = np.ones((5, 5), np.uint8)

        # Iteracja przez faktory i obliczanie najlepszej maski
        for f in factors:
            raw = binarization(self.orig, factor=f)
            mask_opening = opening(raw, kernel)
            mask_opening_closing = closing(mask_opening, kernel)
            mask_opening_closing_llc = largest_connected_component(mask_opening_closing)
            mask = (mask_opening_closing_llc * 255).astype(np.uint8)
            score = circularity_and_completeness(mask)
            
            if score > best_score:
                best_score = score
                best_mask = mask
            print(score, f)

        # Teraz tworzymy okrąg na podstawie najlepszej maski
        proj_y = np.sum(best_mask // 255, axis=1)
        rows = np.where(proj_y > 0)[0]
        proj_x = np.sum(best_mask // 255, axis=0)
        cols = np.where(proj_x > 0)[0]
        
        if rows.size == 0 or cols.size == 0:
            self.pcen = (0, 0)
            self.prad = 0
            return
        
        # Obliczamy środek i promień maski źrenicy
        cy = (rows[0] + rows[-1]) // 2
        ry = (rows[-1] - rows[0]) // 2
        cx = (cols[0] + cols[-1]) // 2
        rx = (cols[-1] - cols[0]) // 2
        self.pcen = (cx, cy)
        self.prad = (ry + rx) // 2

        # --- Tworzenie maski okręgu (czysta NumPy) ---
        rr, cc = np.ogrid[:best_mask.shape[0], :best_mask.shape[1]]
        mask_circle = (rr - cy) ** 2 + (cc - cx) ** 2 <= (self.prad + 5) ** 2  # Margin 5 dla rozszerzenia promienia
        final_mask = np.zeros_like(best_mask, dtype=np.uint8)
        final_mask[mask_circle] = 255  # Maska binarna z okręgiem

        # Przechowywanie ostatecznej maski źrenicy
        self.bin_pupil = final_mask

        # Wizualizacja: rysowanie okręgu na obrazie źrenicy
        self.overlay_pupil = self.orig.copy()
        cv2.circle(self.overlay_pupil, self.pcen, self.prad, (255, 0, 0), 2)  # Rysowanie czerwonego okręgu źrenicy
        self.orig_no_pupil = self.orig.copy()
        rr, cc = np.ogrid[:self.orig.shape[0], :self.orig.shape[1]]
        mask_circle = (rr - cy) ** 2 + (cc - cx) ** 2 <= (self.prad + 5) ** 2  # Używamy promienia źrenicy + margin
        self.orig_no_pupil[mask_circle] = 255 

    def _compute_pupil_overlay(self):
        if self.pcen is None: self._compute_pupil()
        self.overlay_p = self.orig.copy()
        # self.overlay_p = self.bin_pupil.copy()
        cv2.circle(self.overlay_p, self.pcen, int(self.prad), (255, 0, 0), 2)

# ------------- iris processing -------------------------------------- --------------------------------------
    def _compute_iris(self):
        if self.pcen is None: self._compute_pupil()
        
        # Wykorzystaj funkcję detekcji tęczówki od źrenicy
        self.irad = detect_iris_from_pupil(self.gray, self.pcen, self.prad)
        self.ircen = self.pcen  # Zazwyczaj środek tęczówki jest bardzo zbliżony do środka źrenicy
        
        # Tworzenie maski tęczówki
        h, w = self.gray.shape
        yy, xx = np.ogrid[:h, :w]
        
        # Maska pierścieniowa tęczówki (między źrenicą a granicą tęczówki)
        iris_disk = (xx - self.pcen[0])**2 + (yy - self.pcen[1])**2 <= self.irad**2
        pupil_disk = (xx - self.pcen[0])**2 + (yy - self.pcen[1])**2 <= self.prad**2
        iris_ring = iris_disk & (~pupil_disk)
        
        # Stwórz obraz binarny
        self.bin_iris = np.zeros((h, w), dtype=np.uint8)
        self.bin_iris[iris_ring] = 255
        
        # Wizualizacja: rysowanie okręgu na obrazie tęczówki
        self.overlay_iris = self.orig.copy()
        cv2.circle(self.overlay_iris, self.pcen, self.irad, (0, 255, 0), 2)  # Zielony okrąg tęczówki

    def _compute_both_overlay(self):
        if self.irad is None: self._compute_iris()
        self.overlay_both = self.orig.copy()
        cv2.circle(self.overlay_both, self.pcen, int(self.prad), (255, 0, 0), 2)
        cv2.circle(self.overlay_both, self.pcen, int(self.irad), (0, 255, 0), 2)

    # ------------- unwrap ------------------------------------------------
    def _compute_unwrap(self):
        if self.irad is None: self._compute_iris()
        
        # Standardowe wymiary dla unwrappowanej tęczówki
        # Wysokość = odległość między źrenicą a krawędzią tęczówki
        # Szerokość = wystarczająca rozdzielczość dla obwodu
        h_res = min(64, max(32, int(self.irad - self.prad)))  # Między 32 a 64 pikseli
        avg_radius = (self.prad + self.irad) / 2
        w_res = min(512, max(256, int(2 * np.pi * avg_radius)))  # Między 256 a 512 pikseli
        
        # Kąty próbkowania (pełny okrąg)
        thetas = np.linspace(0, 2 * np.pi, w_res, endpoint=False)
        
        # Promienie próbkowania (od źrenicy do tęczówki)
        rs = np.linspace(self.prad, self.irad, h_res)
        
        # Przygotuj pusty obraz kolorowy na unwrappowaną tęczówkę
        # Dla obrazu kolorowego potrzebujemy 3 kanałów
        unwrapped = np.zeros((h_res, w_res, 3), dtype=np.uint8)
        
        # Dla każdego promienia
        for i, r in enumerate(rs):
            # Oblicz współrzędne punktów wzdłuż okręgu dla tego promienia
            x = (r * np.cos(thetas) + self.pcen[0]).astype(int)
            y = (r * np.sin(thetas) + self.pcen[1]).astype(int)
            
            # Sprawdź, które punkty są w granicach obrazu
            valid = (x >= 0) & (x < self.orig.shape[1]) & (y >= 0) & (y < self.orig.shape[0])
            
            # Kopiuj wartości pikseli z oryginalnego obrazu do unwrappowanego
            unwrapped[i, valid] = self.orig[y[valid], x[valid]]
        
        self.unwrapped = unwrapped

# ----------------------------- run --------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv); w = IrisUI(); w.show(); sys.exit(app.exec_())
