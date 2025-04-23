
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
    horizontal_projection, vertical_projection, polar_to_cartesian, close_and_fill, refine_circle_and_score
)

# ---------------- helper -------------------------------------------------

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

    # ------------- pupil processing -------------------------------------
    @staticmethod
    def _largest_connected_component(bin_img):
        """Return mask of the largest 8‑connected component (NumPy only)."""
        h, w = bin_img.shape
        visited = np.zeros_like(bin_img, dtype=bool)
        labels  = np.zeros_like(bin_img, dtype=bool)
        best_sz = 0; best_mask = None
        stack = []
        # iterate over foreground pixels
        for y, x in zip(*np.where(bin_img)):
            if visited[y, x]:
                continue
            cur_mask = []
            stack.append((y, x)); visited[y, x] = True
            while stack:
                cy, cx = stack.pop(); cur_mask.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and bin_img[ny, nx]:
                            visited[ny, nx] = True; stack.append((ny, nx))
            if len(cur_mask) > best_sz:
                best_sz = len(cur_mask)
                best_mask = cur_mask
        mask = np.zeros_like(bin_img, dtype=bool)
        if best_mask is not None:
            ys, xs = zip(*best_mask)
            mask[ys, xs] = True
        return mask

    def _compute_pupil(self):

        best_score = 0
        best_mask = None
        factors = [i/10 for i in range(29, 60, 3)]

        for f in factors:
            raw = binarization(self.orig, factor=f)
            k = np.ones((5, 5), np.uint8)
            mask_opening = opening(raw, k)
            mask_opening_closing = closing(mask_opening, k)
            mask_opening_closing_llc = self._largest_connected_component(mask_opening_closing)
            mask = (mask_opening_closing_llc * 255).astype(np.uint8)
            ideal_mask, score = refine_circle_and_score(mask)
            if score > best_score:
                best_score = score
                best_mask = mask
            print(score, f)

        self.bin_pupil = best_mask

        proj_y = np.sum(mask // 255, axis=1)
        rows = np.where(proj_y > 0)[0]
        proj_x = np.sum(mask // 255, axis=0)
        cols = np.where(proj_x > 0)[0]
        if rows.size == 0 or cols.size == 0:
            self.pcen = (0, 0); self.prad = 0; return
        cy = (rows[0] + rows[-1]) // 2
        ry = (rows[-1] - rows[0]) // 2
        cx = (cols[0] + cols[-1]) // 2
        rx = (cols[-1] - cols[0]) // 2
        self.pcen = (cx, cy)
        self.prad = (ry + rx) // 2

    def _compute_pupil_overlay(self):
        if self.pcen is None: self._compute_pupil()
        self.overlay_p = self.orig.copy()
        # self.overlay_p = self.bin_pupil.copy()
        cv2.circle(self.overlay_p, self.pcen, int(self.prad), (255, 0, 0), 2)

# ------------- iris processing -------------------------------------- --------------------------------------
    def _compute_iris(self):
        if self.pcen is None: self._compute_pupil()
        # estimate iris radius as 3× pupil and threshold on ring only
        est_irad = int(self.prad * 3.0)
        ring_bin = binarization_ring(self.orig, self.pcen, int(self.prad * 1.2), est_irad, factor=2.5)
        # clean lashes via opening-closing using 7×7 kernel
        k = np.ones((5, 5), np.uint8)
        opened = dilatation(erosion(ring_bin, k), k)
        closed = erosion(dilatation(opened, k), k)
        self.bin_iris = closed
        # cy, ry = horizontal_projection(self.bin_iris, None)
        # cx, rx = vertical_projection(self.bin_iris, None)
        # self.irad = (ry + rx) // 2

        proj_y = np.sum(closed // 255, axis=1)
        rows = np.where(proj_y > 0)[0]
        proj_x = np.sum(closed // 255, axis=0)
        cols = np.where(proj_x > 0)[0]
        ry = (rows[-1] - rows[0]) // 2
        rx = (cols[-1] - cols[0]) // 2

        self.irad = (ry + rx) // 2

    def _compute_both_overlay(self):
        if self.irad is None: self._compute_iris()
        self.overlay_both = self.orig.copy()
        cv2.circle(self.overlay_both, self.pcen, int(self.prad), (255, 0, 0), 2)
        cv2.circle(self.overlay_both, self.pcen, int(self.irad), (0, 255, 0), 2)

    # ------------- unwrap ------------------------------------------------
    def _compute_unwrap(self):
        if self.irad is None: self._compute_iris()
        h_res, w_res = 64, 512
        thetas = np.linspace(0, 2 * np.pi, w_res, endpoint=False)
        rs = np.linspace(self.prad, self.irad, h_res)
        un = np.zeros((h_res, w_res), dtype=np.uint8)
        for i, r in enumerate(rs):
            x = (r * np.cos(thetas) + self.pcen[0]).astype(int)
            y = (r * np.sin(thetas) + self.pcen[1]).astype(int)
            valid = (x >= 0) & (x < self.gray.shape[1]) & (y >= 0) & (y < self.gray.shape[0])
            un[i, valid] = self.gray[y[valid], x[valid]]
        self.unwrapped = un


# ----------------------------- run --------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv); w = IrisUI(); w.show(); sys.exit(app.exec_())
