import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QFileDialog, QTabWidget, QLabel, QCheckBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from operations import (
    grayscale, binarization, opening, closing,
    circularity_and_completeness, detect_iris_from_pupil, largest_connected_component, fit_circle_bottom_anchor
)

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

# -------------------------------------- ui --------------------------------------

    def _reset_state(self):
        self.orig = self.gray = None
        self.bin_pupil = self.bin_iris = None
        self.pcen = self.prad = None
        self.irad = None
        self.overlay_p = self.overlay_both = None
        self.unwrapped = None
        self.resize(800, 600)

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        v = QVBoxLayout(c)

        top = QHBoxLayout()
        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.load_image)
        top.addWidget(btn_load)
        self.chk_find_circle = QCheckBox("Find circle")
        self.chk_find_circle.toggled.connect(self._on_find_circle_toggled)
        top.addWidget(self.chk_find_circle)
        v.addLayout(top)

        self.tabs = QTabWidget(); v.addWidget(self.tabs)
        self.labels = []
        for t in self.TAB_TITLES:
            lab = QLabel(alignment=Qt.AlignCenter)
            lab.setScaledContents(False)
            tab = QWidget()
            QVBoxLayout(tab).addWidget(lab)
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

    def update_tab(self, idx):
        if self.orig is None: return
        if idx == 0:
            self._show(0, self.orig)
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

    def _show(self, slot, img):
        pix = cv_to_qpixmap(img).scaled(self.max_w, self.max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lab = self.labels[slot]
        lab.setPixmap(pix)

# -------------------------------------- pupil processing --------------------------------------
    def _compute_pupil(self):

        best_score = 0
        best_mask = None
        factors = [i/10 for i in range(29, 60, 3)]

        for f in factors:
            raw = binarization(self.orig, factor=f)
            k = np.ones((5, 5), np.uint8)
            mask_opening = opening(raw, k)
            mask_opening_closing = closing(mask_opening, k)
            mask_opening_closing_llc = largest_connected_component(mask_opening_closing)
            mask = (mask_opening_closing_llc * 255).astype(np.uint8)
            score = circularity_and_completeness(mask)
            if score > best_score:
                best_score = score
                best_mask = mask
            print(score, f)

        self.bin_pupil = best_mask

        proj_y = np.sum(best_mask // 255, axis=1)
        rows = np.where(proj_y > 0)[0]
        proj_x = np.sum(best_mask // 255, axis=0)
        cols = np.where(proj_x > 0)[0]
        if rows.size == 0 or cols.size == 0:
            self.pcen = (0, 0); self.prad = 0
            return
        cy = (rows[0] + rows[-1]) // 2
        ry = (rows[-1] - rows[0]) // 2
        cx = (cols[0] + cols[-1]) // 2
        rx = (cols[-1] - cols[0]) // 2
        self.pcen = (cx, cy)
        self.prad = (ry + rx) // 2

        # rr, cc = np.ogrid[:best_mask.shape[0], :best_mask.shape[1]]
        # mask_circle = (rr - cy) ** 2 + (cc - cx) ** 2 <= (self.prad + 5) ** 2
        # final_mask = np.zeros_like(best_mask, dtype=np.uint8)
        # final_mask[mask_circle] = 255
        # self.pupil_mask = final_mask

        print(self.chk_find_circle.isChecked())

    def _on_find_circle_toggled(self, checked: bool) -> None:

        if checked and self.bin_pupil.copy() is not None:
            self.tabs.setDisabled(True)

            bin_mask = self.bin_pupil.copy()

            best = fit_circle_bottom_anchor(bin_mask,
                                                 cover_thresh=0.60,
                                                 r_min=100,
                                                 use_perimeter=True,
                                                 return_largest=True)
            if best is None:
                print("No circle found")
            else:
                xc, yc, r = best
                print(f"centre=({xc:.1f},{yc:.1f}),  radius={r}")
                self.pcen = (xc, yc)
                self.prad = r

                if self.tabs.currentIndex() == 2:
                    self._compute_pupil_overlay()
                    self.update_tab(2)

            self.tabs.setEnabled(True)

        else:
            pass

    def _compute_pupil_overlay(self):
        if self.pcen is None: self._compute_pupil()
        self.overlay_p = self.orig.copy()
        # self.overlay_p = self.bin_pupil.copy()
        cv2.circle(self.overlay_p, self.pcen, int(self.prad), (255, 0, 0), 2)

# -------------------------------------- iris processing --------------------------------------
    def _compute_iris(self):
        if self.pcen is None: self._compute_pupil()

        self.irad = detect_iris_from_pupil(self.gray, self.pcen, self.prad)
        self.ircen = self.pcen

        h, w = self.gray.shape
        yy, xx = np.ogrid[:h, :w]

        iris_disk = (xx - self.pcen[0])**2 + (yy - self.pcen[1])**2 <= self.irad**2
        pupil_disk = (xx - self.pcen[0])**2 + (yy - self.pcen[1])**2 <= self.prad**2
        iris_ring = iris_disk & (~pupil_disk)

        self.bin_iris = np.zeros((h, w), dtype=np.uint8)
        self.bin_iris[iris_ring] = 255

        self.overlay_iris = self.orig.copy()
        cv2.circle(self.overlay_iris, self.pcen, self.irad, (0, 255, 0), 2)

    def _compute_both_overlay(self):
        if self.irad is None: self._compute_iris()
        self.overlay_both = self.orig.copy()
        cv2.circle(self.overlay_both, self.pcen, int(self.prad), (255, 0, 0), 2)
        cv2.circle(self.overlay_both, self.pcen, int(self.irad), (0, 255, 0), 2)

# -------------------------------------- unwrap --------------------------------------
    def _compute_unwrap(self):
        if self.irad is None: self._compute_iris()

        h_res = min(64, max(32, int(self.irad - self.prad)))
        avg_radius = (self.prad + self.irad) / 2
        w_res = min(512, max(256, int(2 * np.pi * avg_radius)))

        thetas = np.linspace(0, 2 * np.pi, w_res, endpoint=False)
        rs = np.linspace(self.prad, self.irad, h_res)
        unwrapped = np.zeros((h_res, w_res, 3), dtype=np.uint8)

        for i, r in enumerate(rs):
            x = (r * np.cos(thetas) + self.pcen[0]).astype(int)
            y = (r * np.sin(thetas) + self.pcen[1]).astype(int)
            valid = (x >= 0) & (x < self.orig.shape[1]) & (y >= 0) & (y < self.orig.shape[0])
            unwrapped[i, valid] = self.orig[y[valid], x[valid]]

        self.unwrapped = unwrapped

# ----------------------------- run --------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv); w = IrisUI(); w.show(); sys.exit(app.exec_())