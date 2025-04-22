# main.py
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
    grayscale, binarization, erosion, dilatation,
    horizontal_projection, vertical_projection, polar_to_cartesian
)


def cv_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert an RGB or grayscale NumPy array to QPixmap."""
    if img.ndim == 2:
        h, w = img.shape
        bytes_per_line = w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_Indexed8)
    else:
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class IrisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris Recognition Steps")
        # Determine max display size as quarter of screen
        screen = QApplication.primaryScreen().size()
        self.max_w = screen.width() // 2
        self.max_h = screen.height() // 2

        # Storage for processing
        self.orig = None
        self.gray = None
        self.binary = None
        self.overlay = None
        self.unwrapped = None

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Load button
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(load_btn)
        layout.addLayout(btn_layout)

        # Tabs for steps
        self.tabs = QTabWidget()
        self.step_labels = []
        for name in ["Step 0: Original", "Step 1: Gray/Bin", "Step 2: Borders", "Step 3: Unwrapped"]:
            label = QLabel(alignment=Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setScaledContents(True)
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.addWidget(label)
            self.tabs.addTab(tab, name)
            self.step_labels.append(label)
        self.tabs.currentChanged.connect(self.update_step)
        layout.addWidget(self.tabs)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Eye Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        img = cv2.imread(path)
        self.orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Reset caches
        self.gray = self.binary = self.overlay = self.unwrapped = None
        # Show original at reduced size
        pix = cv_to_qpixmap(self.orig).scaled(self.max_w, self.max_h, Qt.KeepAspectRatio)
        self.step_labels[0].setPixmap(pix)
        self.tabs.setCurrentIndex(0)

    def update_step(self, index):
        if self.orig is None:
            return
        # Step 1: Grayscale & Binarization
        if index == 1:
            if self.gray is None:
                # 1) grayscale
                self.gray = grayscale(self.orig)

                # 2) rough pupil detection on a quick global binarization (to get center & radius)
                init_bin = binarization(self.orig, factor=3.0)
                cy, ry = horizontal_projection(init_bin, factor=3.0)
                cx, rx = vertical_projection(init_bin, factor=3.0)
                pup_center = (cx, cy)
                pup_rad = (ry + rx) // 2

                # 3) ROI‑based binarization of the iris ring using your spec P/X_I
                #    inner radius = pupil radius, outer radius = estimate iris radius
                #    here we use X_I=2.5 and a guessed iris radius (you can refine in Step 2)
                iris_rad = int(pup_rad * 3)  # for example, assume iris ~3× pupil
                from operations import binarization_roi
                self.binary = binarization_roi(
                    self.orig,
                    center=pup_center,
                    inner_r=pup_rad,
                    outer_r=iris_rad,
                    factor=2.5
                )
            # make negative for display
            temp_binary = self.binary.copy()
            self.binary[temp_binary == 255] = 0
            self.binary[temp_binary == 0] = 255

            # show the ROI‑binarized mask
            pix = cv_to_qpixmap(self.binary).scaled(self.max_w, self.max_h, Qt.KeepAspectRatio)
            self.step_labels[1].setPixmap(pix)
        # Step 2: Borders
        elif index == 2:
            if self.overlay is None:
                kernel = np.ones((5, 5), np.uint8)
                bin_iris = dilatation(erosion(self.binary, kernel), kernel)
                bin_pupil = binarization(self.orig, factor=3.0)
                bin_pupil = dilatation(erosion(bin_pupil, kernel), kernel)
                cy_ir, ry_ir = horizontal_projection(bin_iris, factor=2.5)
                cx_ir, rx_ir = vertical_projection(bin_iris, factor=2.5)
                cy_pu, ry_pu = horizontal_projection(bin_pupil, factor=3.0)
                cx_pu, rx_pu = vertical_projection(bin_pupil, factor=3.0)
                cen_ir = (cx_ir, cy_ir)
                cen_pu = (cx_pu, cy_pu)
                rad_ir = (ry_ir + rx_ir) // 2
                rad_pu = (ry_pu + rx_pu) // 2
                self.overlay = self.orig.copy()
                cv2.circle(self.overlay, cen_pu, int(rad_pu), (255, 0, 0), 2)
                cv2.circle(self.overlay, cen_ir, int(rad_ir), (0, 255, 0), 2)
            pix = cv_to_qpixmap(self.overlay).scaled(self.max_w, self.max_h, Qt.KeepAspectRatio)
            self.step_labels[2].setPixmap(pix)
        # Step 3: Unwrapped
        elif index == 3:
            if self.unwrapped is None:
                if self.gray is None:
                    self.gray = grayscale(self.orig)
                kernel = np.ones((5, 5), np.uint8)
                bin_iris = dilatation(erosion(binarization(self.orig, 2.5), kernel), kernel)
                cy_ir, ry_ir = horizontal_projection(bin_iris, factor=2.5)
                cx_ir, rx_ir = vertical_projection(bin_iris, factor=2.5)
                cy_pu, ry_pu = horizontal_projection(binarization(self.orig, 3.0), factor=3.0)
                cx_pu, rx_pu = vertical_projection(binarization(self.orig, 3.0), factor=3.0)
                cen_ir = (cx_ir, cy_ir)
                rad_ir = (ry_ir + rx_ir) // 2
                rad_pu = (ry_pu + rx_pu) // 2
                h_res, w_res = 64, 512
                theta = np.linspace(0, 2 * np.pi, w_res)
                r = np.linspace(rad_pu, rad_ir, h_res)
                self.unwrapped = np.zeros((h_res, w_res), dtype=np.uint8)
                for i, ri in enumerate(r):
                    for j, th in enumerate(theta):
                        x, y = polar_to_cartesian(ri, th, *cen_ir)
                        if 0 <= y < self.gray.shape[0] and 0 <= x < self.gray.shape[1]:
                            self.unwrapped[i, j] = self.gray[y, x]
            pix = cv_to_qpixmap(self.unwrapped).scaled(self.max_w, self.max_h, Qt.KeepAspectRatio)
            self.step_labels[3].setPixmap(pix)


def main():
    app = QApplication(sys.argv)
    window = IrisUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
