import sys
import os
import math
import cv2
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import pandas as pd
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtWidgets import QSplashScreen, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QProgressBar, QMenuBar, QStatusBar, QScrollArea, QSizePolicy
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter
from PyQt6.QtCore import Qt, QTimer
import time # <--- Added this line to import the time module

class ROISelector:
    def __init__(self, img, winname="ROI Selection"):
        self.img = img.copy()
        self.orig = img.copy()
        self.win = winname
        self.roi = None
        self.done = False
        self.reset = False
        self.start = self.end = None
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self.mouse)
        self.instructions = (
            "Draw ROI: drag mouse, Release to finish.\n"
            "'Enter' to confirm, 'q' to quit, 'r' to reset selection."
        )

    def mouse(self, event, x, y, flags, param):
        if self.done or self.reset:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start:
            img = self.orig.copy()
            cv2.rectangle(img, self.start, (x, y), (0,255,0), 2)
            cv2.putText(img, self.instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv2.imshow(self.win, img)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end = (x, y)
            self.done = True

    def get(self):
        img = self.img.copy()
        cv2.putText(img, self.instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        cv2.imshow(self.win, img)
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 13:  # Enter key to confirm
                if self.done and self.start and self.end:
                    x0, y0 = self.start
                    x1, y1 = self.end
                    xa, xb = min(x0,x1), max(x0,x1)
                    ya, yb = min(y0,y1), max(y0,y1)
                    cv2.rectangle(self.img, (xa,ya), (xb,yb), (0,255,0), 2)
                    cv2.putText(self.img, "ROI selected. Press 'Enter' to accept, 'r' to redo, 'q' to quit.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
                    cv2.imshow(self.win, self.img)
                    roi = (xa, ya, xb-xa, yb-ya)
                    # Wait for user to accept, redo, or quit
                    while True:
                        k2 = cv2.waitKey(1) & 0xFF
                        if k2 == 13:  # Enter to accept
                            cv2.destroyWindow(self.win)
                            return roi
                        if k2 == ord("q"):  # q to quit
                            cv2.destroyWindow(self.win)
                            return None
                        if k2 == ord("r"):  # r to redo
                            self.img = self.orig.copy()
                            self.done = False
                            self.start = self.end = None
                            cv2.putText(self.img, self.instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                            cv2.imshow(self.win, self.img)
                            break
                else:
                    # If not done, ignore Enter
                    continue
            if k == ord("q"):  # q to quit
                cv2.destroyWindow(self.win)
                return None
            if k == ord("r"):
                self.start = self.end = None
                self.done = False
                img = self.orig.copy()
                cv2.putText(img, self.instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                cv2.imshow(self.win, img)

class CalibSelector:
    def __init__(self, img, winname="Calibration"):
        self.img = img.copy()
        self.orig = img.copy()
        self.win = winname
        self.pt1 = None
        self.pt2 = None
        self.active = False
        self.instructions = (
            "Draw line for calibration: click-drag.\n"
            "'Enter' to confirm, 'q' to quit, 'r' to reset."
        )
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self.mouse)

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pt1 = (x, y)
            self.pt2 = None
            self.active = True
        elif event == cv2.EVENT_MOUSEMOVE and self.active and self.pt1:
            img = self.orig.copy()
            cv2.line(img, self.pt1, (x, y), (0,255,0), 2)
            cv2.putText(img, self.instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv2.imshow(self.win, img)
        elif event == cv2.EVENT_LBUTTONUP and self.pt1:
            self.pt2 = (x, y)
            self.active = False

    def get(self):
        img = self.img.copy()
        cv2.putText(img, self.instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        cv2.imshow(self.win, img)
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                cv2.destroyWindow(self.win)
                return None
            if k == ord("r"):
                self.pt1 = self.pt2 = None
                img = self.orig.copy()
                cv2.putText(img, self.instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                cv2.imshow(self.win, img)
            if self.pt1 and self.pt2:
                img2 = self.img.copy()
                cv2.line(img2, self.pt1, self.pt2, (0,255,0), 2)
                cv2.putText(img2, "Calibration line done. Press 'Enter' to accept, 'r' to redo, 'q' to quit.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
                cv2.imshow(self.win, img2)
                while True:
                    k2 = cv2.waitKey(1) & 0xFF
                    if k2 == 13:  # Enter key
                        cv2.destroyWindow(self.win)
                        return (self.pt1, self.pt2)
                    if k2 == ord("r"):
                        self.pt1 = self.pt2 = None
                        img = self.orig.copy()
                        cv2.putText(img, self.instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                        cv2.imshow(self.win, img)
                        break

class FluidGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Turbulence Realm - SINDy")
        self.setMinimumSize(800, 800) # Set a minimum size for better layout

        logo_path = "logo.png"
        if os.path.exists(logo_path):
            self.setWindowIcon(QtGui.QIcon(logo_path))

        # Main layout for the entire window
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) # Remove margins for full-width header/footer

        # --- Header (Menu Bar) ---
        self.menu_bar = QMenuBar(self)
        file_menu = self.menu_bar.addMenu("File")
        about_action = QtGui.QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        file_menu.addAction(about_action)
        main_layout.addWidget(self.menu_bar)

        # --- Tab Widget for main content ---
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { /* The tab widget frame */
                border-top: 1px solid #C2C7CB;
                background: white;
            }
            QTabBar::tab {
                background: #e0e0e0;
                border: 1px solid #C2C7CB;
                border-bottom-color: #C2C7CB; /* same as pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 120px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: white;
                border-color: #C2C7CB;
                border-bottom-color: white; /* selected tab has no border on the bottom */
            }
            QTabBar::tab:hover {
                background: #f0f0f0;
            }
        """)
        main_layout.addWidget(self.tabs)

        # --- Tab 1: Setup & Processing ---
        self.setup_tab = QWidget()
        setup_layout = QVBoxLayout(self.setup_tab)
        setup_layout.setContentsMargins(20, 20, 20, 20) # Add padding inside tab

        # File Selection
        file_group_box = QtWidgets.QGroupBox("Video File Selection")
        file_group_layout = QHBoxLayout(file_group_box)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("No video file selected...")
        self.file_btn = QPushButton("Browse video")
        self.file_btn.clicked.connect(self.pick_file)
        file_group_layout.addWidget(self.file_edit)
        file_group_layout.addWidget(self.file_btn)
        setup_layout.addWidget(file_group_box)

        # ROI & Calibration
        roi_calib_group_box = QtWidgets.QGroupBox("Region of Interest & Calibration")
        roi_calib_layout = QVBoxLayout(roi_calib_group_box)
        self.run_btn = QPushButton("ROI & Calibration")
        self.run_btn.clicked.connect(self.start_roi)
        self.mpp_label = QLabel("Calibration: -- px = -- m; 1 px = -- m")
        roi_calib_layout.addWidget(self.run_btn)
        roi_calib_layout.addWidget(self.mpp_label)
        setup_layout.addWidget(roi_calib_group_box)

        # Processing Steps
        processing_group_box = QtWidgets.QGroupBox("Processing Steps")
        processing_layout = QGridLayout(processing_group_box)

        self.process_btn = QPushButton("Process Optical Flow")
        self.process_btn.clicked.connect(self.run_optical_flow_gui)
        self.process_btn.setEnabled(False)
        processing_layout.addWidget(self.process_btn, 0, 0)

        self.sindy_btn = QPushButton("Run SINDy Modeling")
        self.sindy_btn.clicked.connect(self.run_sindy_gui)
        self.sindy_btn.setEnabled(False)
        processing_layout.addWidget(self.sindy_btn, 0, 1)

        self.pred_btn = QPushButton("Run SINDy Prediction")
        self.pred_btn.clicked.connect(self.run_prediction_gui)
        self.pred_btn.setEnabled(False)
        processing_layout.addWidget(self.pred_btn, 1, 0)

        self.sindy_eq_btn = QPushButton("Show SINDy Equation")
        self.sindy_eq_btn.clicked.connect(self.show_sindy_equation_gui)
        self.sindy_eq_btn.setEnabled(False)
        processing_layout.addWidget(self.sindy_eq_btn, 1, 1)

        setup_layout.addWidget(processing_group_box)

        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setMaximum(100)
        self.progress.hide() # Hide initially
        setup_layout.addWidget(self.progress)

        # Status Label for step-by-step instructions
        self.status = QLabel("1. Choose video file.\n2. ROI & Calibration\n3. Process Optical Flow\n4. SINDy Fit\n5. Prediction, Analysis, Export.\n")
        self.status.setStyleSheet("QLabel { color: #34495e; font-style: italic; padding: 10px; border: 1px solid #dcdcdc; border-radius: 5px; background-color: #f8f8f8; }")
        setup_layout.addWidget(self.status)

        setup_layout.addStretch(1) # Push content to top
        self.tabs.addTab(self.setup_tab, "Setup & Processing")

        # --- Tab 2: Visualization & Analysis ---
        self.visualize_tab = QWidget()
        visualize_layout = QVBoxLayout(self.visualize_tab)
        visualize_layout.setContentsMargins(20, 20, 20, 20)

        # Visualization Buttons
        viz_buttons_group_box = QtWidgets.QGroupBox("Visualization & Analysis Tools")
        viz_buttons_layout = QGridLayout(viz_buttons_group_box)

        self.quiver_btn = QPushButton("Quiver Comparison")
        self.quiver_btn.clicked.connect(self.show_quiver_section)
        self.quiver_btn.setEnabled(False)
        viz_buttons_layout.addWidget(self.quiver_btn, 0, 0)

        self.contour_btn = QPushButton("Contour Plot")
        self.contour_btn.clicked.connect(self.plot_contour_gui)
        self.contour_btn.setEnabled(False)
        viz_buttons_layout.addWidget(self.contour_btn, 0, 1)

        self.stream_btn = QPushButton("Streamline Plot")
        self.stream_btn.clicked.connect(self.plot_stream_gui)
        self.stream_btn.setEnabled(False)
        viz_buttons_layout.addWidget(self.stream_btn, 1, 0)

        self.error_btn = QPushButton("Error Analysis/Maps")
        self.error_btn.clicked.connect(self.run_error_analysis_gui)
        self.error_btn.setEnabled(False)
        viz_buttons_layout.addWidget(self.error_btn, 1, 1)

        visualize_layout.addWidget(viz_buttons_group_box)

        # Matplotlib Figure Area (Quiver Comparison section integrated here)
        self.quiver_section = QWidget() # This will now be part of the visualize_tab
        quiver_layout = QVBoxLayout(self.quiver_section)

        # Slider components
        slider_layout = QHBoxLayout()
        self.quiver_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.quiver_slider.setMinimum(1)
        self.quiver_slider.setMaximum(1) # Will be updated dynamically
        self.quiver_frame_label = QLabel("Frame: 1/1")
        slider_layout.addWidget(QLabel("Frame:"))
        slider_layout.addWidget(self.quiver_slider)
        slider_layout.addWidget(self.quiver_frame_label)
        quiver_layout.addLayout(slider_layout)

        # Matplotlib figure setup with QWidget container
        self.quiver_container = QWidget()
        container_layout = QVBoxLayout(self.quiver_container)
        self.quiver_fig = plt.Figure(figsize=(8, 6), tight_layout=True)
        self.quiver_ax = self.quiver_fig.add_subplot(111)
        self.quiver_canvas = FigureCanvas(self.quiver_fig)
        container_layout.addWidget(self.quiver_canvas)

        # Add navigation toolbar
        self.quiver_toolbar = NavigationToolbar(self.quiver_canvas, self.quiver_section)
        quiver_layout.addWidget(self.quiver_container)
        quiver_layout.addWidget(self.quiver_toolbar)

        self.quiver_section.setVisible(False) # Hide initially
        visualize_layout.addWidget(self.quiver_section)
        visualize_layout.addStretch(1)
        self.tabs.addTab(self.visualize_tab, "Visualization & Analysis")

        # --- Tab 3: Export Options ---
        self.export_tab = QWidget()
        export_layout = QVBoxLayout(self.export_tab)
        export_layout.setContentsMargins(20, 20, 20, 20)

        export_group_box = QtWidgets.QGroupBox("Export Data & Animations")
        export_buttons_layout = QGridLayout(export_group_box)

        self.export_btn = QPushButton("Export All Frame CSVs")
        self.export_btn.clicked.connect(self.export_csv_gui)
        self.export_btn.setEnabled(False)
        export_buttons_layout.addWidget(self.export_btn, 0, 0)

        self.anim_btn = QPushButton("Export Quiver Animation")
        self.anim_btn.clicked.connect(self.export_animation_gui)
        self.anim_btn.setEnabled(False)
        export_buttons_layout.addWidget(self.anim_btn, 0, 1)

        # self.stream_anim_btn = QPushButton("Export Streamline Animation")
        # self.stream_anim_btn.clicked.connect(self.export_streamline_animation_gui)
        # self.stream_anim_btn.setEnabled(False)
        # export_buttons_layout.addWidget(self.stream_anim_btn, 1, 0)

        export_layout.addWidget(export_group_box)
        export_layout.addStretch(1)
        self.tabs.addTab(self.export_tab, "Export Options")

        # --- Footer (Status Bar) ---
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)

        # --- Connect button states ---
        self.run_btn.clicked.connect(lambda: self.process_btn.setEnabled(True))
        self.process_btn.clicked.connect(lambda: self.sindy_btn.setEnabled(True))
        self.sindy_btn.clicked.connect(lambda: self.pred_btn.setEnabled(True))
        self.sindy_btn.clicked.connect(lambda: self.sindy_eq_btn.setEnabled(True))
        self.pred_btn.clicked.connect(lambda: self.quiver_btn.setEnabled(True))
        self.pred_btn.clicked.connect(lambda: self.contour_btn.setEnabled(True))
        self.pred_btn.clicked.connect(lambda: self.stream_btn.setEnabled(True))
        self.pred_btn.clicked.connect(lambda: self.export_btn.setEnabled(True))
        self.pred_btn.clicked.connect(lambda: self.error_btn.setEnabled(True))
        self.pred_btn.clicked.connect(lambda: self.anim_btn.setEnabled(True))
        #self.pred_btn.clicked.connect(lambda: self.stream_anim_btn.setEnabled(True))

        # Initial data structures
        self.result = {}
        self.of_result = {}
        self.sindy_result = {}
        self.finished = False
        self.frames_to_display = 8
        self.quiver_step = None

        # Apply general styling
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 14px;
                color: #333;
            }
            QPushButton {
                background-color: #4CAF50; /* Green */
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #39843c;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QGroupBox {
                font-weight: bold;
                margin-top: 10px;
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                padding-top: 15px;
                margin-bottom: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                margin-left: 2px;
                color: #2c3e50;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #007bff; /* Blue */
                border-radius: 5px;
            }
            QStatusBar {
                background-color: #2c3e50; /* Dark blue/grey */
                color: white;
                padding: 5px;
                font-size: 12px;
            }
        """)

    def show_quiver_section(self):
        # Get parameters first
        scale, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver Scale", "Set quiver scale (higher=smaller arrows):",
            1, 0.1, 150, 2
        )
        if not ok: return
        width, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver Width", "Set arrow width:",
            0.006, 0.0001, 0.1, 4
        )
        if not ok: return

        self.quiver_scale = scale
        self.quiver_width = width

        # Initialize slider parameters
        nfr = self.sindy_result['n_frames']
        self.quiver_slider.setMinimum(1)
        self.quiver_slider.setMaximum(nfr)
        self.quiver_slider.setValue(1)
        self.quiver_frame_label.setText(f"Frame: 1/{nfr}")

        # Connect slider if not already connected
        if not hasattr(self, '_quiver_slider_connected'):
            self.quiver_slider.valueChanged.connect(self.update_quiver_plot)
            self._quiver_slider_connected = True

        self.quiver_section.setVisible(True)
        self.update_quiver_plot(1)  # Initial plot

    def update_quiver_plot(self, frame_idx):
        # Validate inputs
        if 'frame_mmap_path' not in self.of_result:
            QtWidgets.QMessageBox.critical(self, "Data Error", "Missing frame data - rerun optical flow processing")
            return

        idx = frame_idx - 1  # Convert to 0-based index
        nfr = self.sindy_result['n_frames']
        if idx < 0 or idx >= nfr:
            return

        try:
            # Load data sources
            meters_per_pixel = self.result['meters_per_pixel']
            u_path = self.of_result['u_mmap_path']
            v_path = self.of_result['v_mmap_path']
            frame_path = self.of_result['frame_mmap_path']

            # Memory map access
            shape = (nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w'])
            u_mmap = np.memmap(u_path, dtype=np.float32, mode='r', shape=shape)
            v_mmap = np.memmap(v_path, dtype=np.float32, mode='r', shape=shape)
            frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r', shape=shape)

            # Get actual frame data
            roi_height, roi_width = u_mmap.shape[1], u_mmap.shape[2]
            bg_img = np.asarray(frame_mmap[idx])
            actual_u = np.asarray(u_mmap[idx])
            actual_v = np.asarray(v_mmap[idx])
            pred_u = self.X_pred_optical[idx, ..., 0]
            pred_v = self.X_pred_optical[idx, ..., 1]

            # Grid setup
            quiver_step = max(roi_height // 25, 2)
            ys = np.arange(0, roi_height, quiver_step)
            xs = np.arange(0, roi_width, quiver_step)
            xg, yg = np.meshgrid(xs, ys)
            xg_real = xg * meters_per_pixel
            yg_real = yg * meters_per_pixel

            # Clear and redraw plot
            self.quiver_ax.clear()
            self.quiver_ax.imshow(bg_img, cmap='gray', origin='upper',
                                extent=[0, roi_width*meters_per_pixel, roi_height*meters_per_pixel, 0])

            # Plot vectors
            self.quiver_ax.quiver(
                xg_real, yg_real,
                actual_u[ys[:, None], xs[None, :]],
                actual_v[ys[:, None], xs[None, :]],
                color='b', angles='xy',
                scale=self.quiver_scale,
                width=self.quiver_width,
                label='Actual'
            )
            self.quiver_ax.quiver(
                xg_real, yg_real,
                pred_u[ys[:, None], xs[None, :]],
                pred_v[ys[:, None], xs[None, :]],
                color='r', angles='xy',
                scale=self.quiver_scale,
                width=self.quiver_width,
                label='SINDy'
            )

            # Configure plot
            self.quiver_ax.set_title(f"Frame {frame_idx}: Actual vs SINDy Prediction")
            self.quiver_ax.set_xlabel("x (m)")
            self.quiver_ax.set_ylabel("y (m)")
            self.quiver_ax.legend()
            self.quiver_ax.set_xlim(0, roi_width*meters_per_pixel)
            self.quiver_ax.set_ylim(roi_height*meters_per_pixel, 0)

            # Adjust layout
            self.quiver_fig.subplots_adjust(
                left=0.08, right=0.95,
                bottom=0.1, top=0.9
            )

            # Update display
            self.quiver_fig.tight_layout()
            # Force full GUI update
            self.quiver_canvas.draw_idle()
            self.quiver_container.updateGeometry()
            QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.processEvents()


        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Plot Error",
                f"Failed to update quiver plot:\n{str(e)}"
            )

    def show_about_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Turbulence Realm - SINDy")
        dlg.setFixedSize(400, 400)  # More compact size

        vbox = QtWidgets.QVBoxLayout(dlg)

        # Logo with proper scaling
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            logo_label = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(logo_path)
            pixmap = pixmap.scaled(100, 100,
                                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                QtCore.Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(logo_label)

        # Title with styling
        title = QtWidgets.QLabel("Turbulence Realm - SINDy")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
        """)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(title)

        # Clean info text
        info = QtWidgets.QLabel("""<p style='text-align: center'>
            Version 1.0<br>
            <em>Video-based Fluid Flow Analysis Tool</em>
        </p>
        <p style='text-align: center; font-size: 11px'>
            Optical Flow & SINDy Modeling based on pysindy<br>
            Developed by Fayaz Rasheed<br>www.turbulencerealm.com<br>
        </p>""")

        info.setOpenExternalLinks(True)
        vbox.addWidget(info)

        # Centered OK button
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dlg.accept)
        vbox.addWidget(btn_box, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        dlg.exec()

    def pick_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video file", ".", "Videos (*.mp4 *.avi *.mov);;All (*)")
        if fname:
            self.file_edit.setText(fname)

    def start_roi(self):
        path = self.file_edit.text()
        if not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self, "No file", "Please select a valid video file first.")
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Open error", "Cannot open selected video.")
            return
        ret, first_frame = cap.read()
        if not ret:
            QtWidgets.QMessageBox.warning(self, "Read error", "Cannot read first frame.")
            cap.release()
            return
        self.status.setText("Draw ROI with mouse. 'q': finish, 'r': reset.")
        roi_selector = ROISelector(first_frame,"ROI Selection")
        roi = roi_selector.get()
        if roi is None:
            self.status.setText("ROI selection cancelled.")
            cap.release()
            return
        xa, ya, roi_w, roi_h = roi
        xb, yb = xa+roi_w, ya+roi_h
        roi_img = first_frame[ya:yb, xa:xb]
        self.status.setText("Draw calibration line. 'q': finish, 'r': reset.")
        calib_selector = CalibSelector(roi_img,"Calibration")
        pts = calib_selector.get()
        if pts is None or pts[0] is None or pts[1] is None:
            self.status.setText("Calibration cancelled.")
            cap.release()
            return
        px_len = np.linalg.norm(np.array(pts[0])-np.array(pts[1]))
        use_meters, ok = QtWidgets.QInputDialog.getDouble(
            self, "Real length", "Enter real-world length (meters) of calibration line:",
            value=0.1, min=1e-6, decimals=6
        )
        if not ok or use_meters <= 0:
            self.status.setText("Calibration cancelled (invalid input).")
            cap.release()
            return
        meters_per_pixel = use_meters / px_len
        self.mpp_label.setText(f"Calibration: {px_len:.2f} px = {use_meters:.5f} m; 1 px = {meters_per_pixel:.8f} m")
        self.result = {
            'video_file': path,
            'roi_box':  (xa, ya, xb, yb),
            'calibration_px': px_len,
            'calibration_m': use_meters,
            'meters_per_pixel': meters_per_pixel,
            'first_frame_roi': roi_img.copy()
        }
        self.finished = True
        self.status.setText("ROI/cali complete. Ready for Optical Flow.")

    def run_optical_flow_gui(self):
        if not getattr(self, 'finished', False):
            QtWidgets.QMessageBox.warning(self, "No ROI/Calibration", "Finish ROI/Calibration first.")
            return
        filters, ok = QtWidgets.QInputDialog.getItem(
            self, "Filtering",
            "Select filter for flow (will preview updates):",
            ("None", "Gaussian Blur", "Nonlocal Means", "Both"), 0, False
        )
        if not ok: return
        gaussian_params = {} ; nlm_params = {}
        enable_gauss = enable_nlm = False
        if filters in ("Gaussian Blur", "Both"):
            enable_gauss = True
            k,ok = QtWidgets.QInputDialog.getInt(self, "Gaussian kernel", "Kernel size (odd)", 5, 3,25,2)
            if not ok: return
            s,ok = QtWidgets.QInputDialog.getDouble(self,"Gaussian sigma","Sigma",1.3,0.01,10.0,2)
            if not ok: return
            gaussian_params = {"ksize": (k,k), "sigmaX": s}
        if filters in ("Nonlocal Means", "Both"):
            enable_nlm = True
            h,ok = QtWidgets.QInputDialog.getDouble(self,"NLM h","h (denoise strength)",10,1,30,1)
            if not ok: return
            tpl, ok = QtWidgets.QInputDialog.getInt(self,"Template Size", "templateWindowSize (odd)",7,3,25,2)
            if not ok: return
            sw, ok = QtWidgets.QInputDialog.getInt(self,"Search Size", "searchWindowSize (odd)",21,5,41,2)
            if not ok: return
            nlm_params = {"h":h, "templateWindowSize":tpl, "searchWindowSize":sw}
        self.status.setText("Optical flow processing started.")
        self.progress.show(); self.progress.setValue(0)
        QtWidgets.QApplication.processEvents()
        box = self.result['roi_box']
        meters_per_pixel = self.result['meters_per_pixel']
        video_file = self.result['video_file']
        MMAP_DIR = "./velocity_mmaps"
        os.makedirs(MMAP_DIR, exist_ok=True)
        u_path = os.path.abspath(os.path.join(MMAP_DIR, "u.dat"))
        v_path = os.path.abspath(os.path.join(MMAP_DIR, "v.dat"))
        for p in [u_path,v_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

        # Frame storage
        frame_path = os.path.join(MMAP_DIR, "frames.dat")
        if os.path.exists(frame_path): os.remove(frame_path)

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        roi_w = box[2]-box[0]
        roi_h = box[3]-box[1]
        n_frames = total_frames-1
        u_mmap = np.memmap(u_path, dtype=np.float32, mode='w+', shape=(n_frames,roi_h,roi_w))
        v_mmap = np.memmap(v_path, dtype=np.float32, mode='w+', shape=(n_frames,roi_h,roi_w))
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='w+', shape=(n_frames, roi_h, roi_w))
        self.of_result["frame_mmap_path"] = frame_path  # Critical for later access

        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2GRAY)
        if enable_gauss:
            prev_gray = cv2.GaussianBlur(prev_gray, **gaussian_params)
        if enable_nlm:
            prev_gray = cv2.fastNlMeansDenoising(prev_gray, None, **nlm_params)
        FPS = cap.get(cv2.CAP_PROP_FPS)
        step_show = max(1, n_frames//40)
        grid_step = max(roi_h//25, 2)
        ygrid = np.arange(0, roi_h, grid_step)
        xgrid = np.arange(0, roi_w, grid_step)
        xgrid2d, ygrid2d = np.meshgrid(xgrid, ygrid)
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret: break
            curr_gray = cv2.cvtColor(frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2GRAY)
            if enable_gauss:
                curr_gray = cv2.GaussianBlur(curr_gray, **gaussian_params)
            if enable_nlm:
                curr_gray = cv2.fastNlMeansDenoising(curr_gray, None, **nlm_params)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, pyr_scale=0.3, levels=7, winsize=21, iterations=7,
                poly_n=7, poly_sigma=1.1, flags=0
            )
            u_mmap[i] = flow[...,0] * meters_per_pixel * FPS
            v_mmap[i] = flow[...,1] * meters_per_pixel * FPS
            frame_mmap[i] = curr_gray  # Store processed frame
            frame_mmap.flush()

            prev_gray = curr_gray.copy()
            if i%step_show==0 or i==n_frames-1:
                rgb_disp = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
                mag,ang = cv2.cartToPolar(flow[...,0],flow[...,1])
                hsv = np.zeros_like(rgb_disp)
                hsv[...,1]=255
                hsv[...,0]=ang*180/np.pi/2
                hsv[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
                preview = cv2.addWeighted(rgb_disp,0.5,bgr,0.5,0).astype(np.uint8)
                # Overlay quiver
                uf = flow[...,0][ygrid2d,xgrid2d]
                vf = flow[...,1][ygrid2d,xgrid2d]
                for (x, y, du, dv) in zip(xgrid2d.flatten(), ygrid2d.flatten(), uf.flatten(), vf.flatten()):
                    tip = (int(round(x+du*2)), int(round(y+dv*2)))
                    cv2.arrowedLine(preview, (int(x),int(y)), tip, (0,0,255), 1, tipLength=0.28)
                inst = "Quiver: Red. Enter: confirm, ESC/q: quit preview."
                cv2.putText(preview, inst, (8,22), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                cv2.imshow("Optical Flow Progress",preview)
                k = cv2.waitKey(1)&0xFF
                if k in (13, 27, ord('q')):  # Enter, ESC, or q
                    break
            if i%3==0:
                self.progress.setValue(int((i+1)/n_frames*100))
                QtWidgets.QApplication.processEvents()
        u_mmap.flush(); v_mmap.flush(); cap.release()
        self.progress.setValue(100); self.progress.hide()
        try: cv2.destroyWindow("Optical Flow Progress")
        except: pass
        self.status.setText(f"Optical flow done [{n_frames}, {roi_h}x{roi_w}]. Run SINDy model next.")
        QtWidgets.QMessageBox.information(self, "Done", "Optical flow complete!")
        self.of_result = {
            "u_mmap_path": u_path, "v_mmap_path": v_path, "frame_mmap_path": frame_path,
            "frames": n_frames, "roi_h": roi_h, "roi_w": roi_w, "FPS":FPS,
            "filters": filters, "enable_gauss": enable_gauss, "gauss": gaussian_params,
            "enable_nlm": enable_nlm, "nlm": nlm_params
        }


    # ------ The rest of the pipeline follows untouched from the previous code, but with prompt index defaults fixed ------
    # ADDITIONAL CHANGES ONLY TO USER PROMPT DEFAULTS AND plt.tight_layout() CALLS

    def run_sindy_gui(self):
        degree, ok = QtWidgets.QInputDialog.getInt(
            self, "SINDy Feature Library", "Polynomial degree (1-5)", 3, 1, 5, 1)
        if not ok: return
        threshold, ok = QtWidgets.QInputDialog.getDouble(
            self, "STLSQ Threshold", "STLSQ threshold", 0.07, 0.01, 1.0, 3)
        if not ok: return
        u_path = self.of_result['u_mmap_path']
        v_path = self.of_result['v_mmap_path']
        n_frames = self.of_result['frames']
        roi_h = self.of_result['roi_h']
        roi_w = self.of_result['roi_w']
        MMAP_DIR = os.path.dirname(u_path)
        u_mmap = np.memmap(u_path, dtype=np.float32, mode='r', shape=(n_frames,roi_h,roi_w))
        v_mmap = np.memmap(v_path, dtype=np.float32, mode='r', shape=u_mmap.shape)
        total_points = n_frames * roi_h * roi_w

        def mmap_gradient(arr_mmap, axis=0):
            g = np.zeros_like(arr_mmap)
            if axis == 0:
                g[1:-1] = (arr_mmap[2:] - arr_mmap[:-2]) / 2.0
                g[0] = arr_mmap[1] - arr_mmap[0]
                g[-1] = arr_mmap[-1] - arr_mmap[-2]
            elif axis == 1:
                g[:, 1:-1, :] = (arr_mmap[:, 2:, :] - arr_mmap[:, :-2, :]) / 2.0
                g[:, 0, :] = arr_mmap[:, 1, :] - arr_mmap[:, 0, :]
                g[:, -1, :] = arr_mmap[:, -1, :] - arr_mmap[:, -2, :] # Corrected slicing
            elif axis == 2:
                g[:, :, 1:-1] = (arr_mmap[:, :, 2:] - arr_mmap[:, :, :-2]) / 2.0
                g[:, :, 0] = arr_mmap[:, :, 1] - arr_mmap[:, :, 0]
                g[:, :, -1] = arr_mmap[:, :, -1] - arr_mmap[:, :, -2] # Corrected slicing
            return g

        self.status.setText("Computing derivatives...")
        QtWidgets.QApplication.processEvents()
        u_x = mmap_gradient(u_mmap, axis=2)
        u_y = mmap_gradient(u_mmap, axis=1)
        u_t = mmap_gradient(u_mmap, axis=0)
        v_x = mmap_gradient(v_mmap, axis=2)
        v_y = mmap_gradient(v_mmap, axis=1)
        v_t = mmap_gradient(v_mmap, axis=0)
        u_xy = mmap_gradient(u_y, axis=2)
        v_xy = mmap_gradient(v_y, axis=2)
        def mmap_stack_X_data(filename, dtype, shape):
            if os.path.exists(filename):
                os.remove(filename)
            return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
        X_optical_path = os.path.join(MMAP_DIR, "X_optical.dat")
        X_dot_optical_path = os.path.join(MMAP_DIR, "X_dot_optical.dat")
        X_optical_mmap = mmap_stack_X_data(X_optical_path, np.float32, (total_points, 8))
        X_dot_optical_mmap = mmap_stack_X_data(X_dot_optical_path, np.float32, (total_points, 2))
        self.status.setText("Preprocessing SINDy input data..."); QtWidgets.QApplication.processEvents()
        idx = 0
        for f in range(n_frames):
            Xf = np.stack([
                u_mmap[f], v_mmap[f], u_x[f], u_y[f], v_x[f], v_y[f], u_xy[f], v_xy[f]
            ], axis=-1)
            Xdf = np.stack([u_t[f], v_t[f]], axis=-1)
            nrow = Xf.shape[0] * Xf.shape[1]
            X_optical_mmap[idx:idx+nrow] = Xf.reshape(-1, 8)
            X_dot_optical_mmap[idx:idx+nrow] = Xdf.reshape(-1, 2)
            idx += nrow
            if f%3==0:
                self.progress.setValue(int((f+1)/n_frames*50))
                QtWidgets.QApplication.processEvents()
        X_optical_mmap.flush(); X_dot_optical_mmap.flush()
        self.status.setText("Fitting SINDy model..."); QtWidgets.QApplication.processEvents()
        batch_size = max(100_000, roi_w * roi_h)
        n_batches = math.ceil(total_points / batch_size)
        DT = 1.0/self.of_result['FPS']
        library = PolynomialLibrary(degree=degree)
        optimizer = STLSQ(threshold=threshold)
        sindy_model = ps.SINDy(feature_library=library, optimizer=optimizer)
        for b in range(n_batches):
            b0 = b*batch_size
            b1 = min(b0+batch_size, total_points)
            if b==0:
                sindy_model.fit(X_optical_mmap[b0:b1], t=DT, x_dot=X_dot_optical_mmap[b0:b1])
            self.progress.setValue(50 + int((b+1)/n_batches*45))
            QtWidgets.QApplication.processEvents()
        self.progress.setValue(95)
        self.status.setText("SINDy fit complete."); QtWidgets.QApplication.processEvents()
        self.sindy_result = {
            'model': sindy_model, 'X_optical_path': X_optical_path, 'X_dot_optical_path': X_dot_optical_path,
            'total_points': total_points, 'n_frames': n_frames, 'roi_h': roi_h, 'roi_w': roi_w, 'batch_size': batch_size,
            'DT': DT, 'library_degree': degree, 'stlsq_thresh': threshold
        }
        QtWidgets.QMessageBox.information(self, "Done", "SINDy model fit Successful!")

    def show_sindy_equation_gui(self):
        sindy_model = self.sindy_result.get('model', None)
        if sindy_model is None:
            QtWidgets.QMessageBox.warning(self, "No SINDy Model", "Please run SINDy modeling first.")
            return
        try:
            # Get current frame index from slider
            current_frame = self.quiver_slider.value()

            eq_text = "\n".join([
                f"Frame {current_frame} Equations:",
                *sindy_model.equations(precision=5)
            ])

            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(f"SINDy Equations - Frame {current_frame}")
            dlg.resize(600, 400)
            vbox = QtWidgets.QVBoxLayout(dlg)
            text = QtWidgets.QPlainTextEdit()
            text.setPlainText(eq_text)
            text.setReadOnly(True)
            vbox.addWidget(text)

            copy_btn = QtWidgets.QPushButton("Copy to Clipboard")
            def copy_to_clipboard():
                QtWidgets.QApplication.clipboard().setText(eq_text)
            copy_btn.clicked.connect(copy_to_clipboard)
            vbox.addWidget(copy_btn)

            dlg.exec()

        except Exception as e:  # Add missing except block
            QtWidgets.QMessageBox.critical(self, "Equation Error",
                f"Failed to display equations:\n{str(e)}")

    def run_prediction_gui(self):
        sindy_model = self.sindy_result['model']
        X_optical_path = self.sindy_result['X_optical_path']
        total_points = self.sindy_result['total_points']
        n_frames = self.sindy_result['n_frames']
        roi_h = self.sindy_result['roi_h']
        roi_w = self.sindy_result['roi_w']
        batch_size = self.sindy_result['batch_size']
        pred_result_path = os.path.join(os.path.dirname(X_optical_path), "X_pred_optical.dat")
        if os.path.exists(pred_result_path): os.remove(pred_result_path)
        X_pred_optical_mmap = np.memmap(pred_result_path, np.float32, mode='w+', shape=(total_points,2))
        n_batches = int(np.ceil(total_points / batch_size))
        self.progress.setValue(0); self.progress.show()
        self.status.setText("Batch prediction (SINDy)..."); QtWidgets.QApplication.processEvents()
        for b in range(n_batches):
            b0 = b*batch_size
            b1 = min((b+1)*batch_size, total_points)
            chunk = np.memmap(X_optical_path, np.float32, mode='r', shape=(total_points,8))[b0:b1]
            pred = sindy_model.predict(chunk)
            pred *= -1.0
            X_pred_optical_mmap[b0:b1] = pred
            self.progress.setValue(int((b+1)*80/n_batches))
            QtWidgets.QApplication.processEvents()
        X_pred_optical_mmap.flush()
        self.status.setText("Prediction done. Postprocessing for plots."); QtWidgets.QApplication.processEvents()
        X_pred_optical = np.zeros((n_frames, roi_h, roi_w, 2), np.float32)
        for f in range(n_frames):
            out = X_pred_optical_mmap[f*roi_h*roi_w:(f+1)*roi_h*roi_w]
            X_pred_optical[f,:,:,0] = gaussian_filter(out[:,0].reshape((roi_h,roi_w)),sigma=0.8)
            X_pred_optical[f,:,:,1] = gaussian_filter(out[:,1].reshape((roi_h,roi_w)),sigma=0.8)
            if f % max(n_frames//10,1) == 0:
                self.progress.setValue(80+int(f/n_frames*20))
                QtWidgets.QApplication.processEvents()
        self.status.setText("Prediction ready. You may now plot/export."); self.progress.setValue(100); QtWidgets.QApplication.processEvents()
        self.X_pred_optical = X_pred_optical
        QtWidgets.QMessageBox.information(self, "Ready", "SINDy prediction and reconstruction ready.")

    def plot_quiver_gui(self):
        # Frame index defaults to 1, displayed as 1-index
        nfr = self.sindy_result['n_frames']
        idx, ok = QtWidgets.QInputDialog.getInt(
            self, "Frame", f"Frame index (1 - {nfr}):", 1, 1, nfr, 1)
        if not ok: return
        idx = idx-1
        scale, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver scale", "Set quiver scale (increase for shorter arrows):", 1, 0.1, 150, 2)
        if not ok: return
        meters_per_pixel = self.result['meters_per_pixel']
        roi_img = self.result['first_frame_roi']
        u_path, v_path = self.of_result['u_mmap_path'], self.of_result['v_mmap_path']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr,self.sindy_result['roi_h'],self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        u, v = np.asarray(u_mmap[idx]), np.asarray(v_mmap[idx])
        xg, yg = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))

        # Get actual frame's ROI from memory map
        frame_path = self.of_result['frame_mmap_path']
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r',
                            shape=(self.sindy_result['n_frames'],
                                    self.sindy_result['roi_h'],
                                    self.sindy_result['roi_w']))
        bg_img = np.asarray(frame_mmap[idx])  # idx is 0-based

        plt.figure(figsize=(7,7))
        plt.imshow(bg_img, cmap='gray', origin='upper', extent=[0,u.shape[1]*meters_per_pixel,u.shape[0]*meters_per_pixel,0])
        plt.quiver(xg*meters_per_pixel, yg*meters_per_pixel, u, v, color='b', scale=scale, width=0.006, label='Actual')
        u_pred = self.X_pred_optical[idx,:,:,0]
        v_pred = self.X_pred_optical[idx,:,:,1]
        plt.quiver(xg*meters_per_pixel, yg*meters_per_pixel, u_pred, v_pred, color='r', scale=scale, width=0.006, label='SINDy')
        plt.title(f"Frame {idx+1}: Quiver Overlay (Blue=Actual, Red=SINDy)")
        plt.xlabel("x (m)"); plt.ylabel("y (m)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_contour_gui(self):
        nfr = self.sindy_result['n_frames']
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Frame", f"Frame index (1 - {nfr}):", 1, 1, nfr, 1)
        if not ok: return
        idx = idx-1
        which, ok = QtWidgets.QInputDialog.getItem(
            self, "Plot Field", "Display contour for:", ("Actual", "SINDy Prediction", "Error"), 0, False)
        if not ok: return
        meters_per_pixel = self.result['meters_per_pixel']
        roi_img = self.result['first_frame_roi']
        u_path, v_path = self.of_result['u_mmap_path'], self.of_result['v_mmap_path']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr,self.sindy_result['roi_h'],self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        u = np.asarray(u_mmap[idx])
        v = np.asarray(v_mmap[idx])
        u_pred = self.X_pred_optical[idx,:,:,0]
        v_pred = self.X_pred_optical[idx,:,:,1]
        if which=="Actual":
            mag = np.sqrt(u**2+v**2)
            title = f"Frame {idx+1} Actual Velocity Magnitude"
        elif which=="SINDy Prediction":
            mag = np.sqrt(u_pred**2+v_pred**2)
            title = f"Frame {idx+1} SINDy Prediction Magnitude"
        else:
            mag = np.sqrt((u-u_pred)**2 + (v-v_pred)**2)
            title = f"Frame {idx+1} Prediction Error Magnitude"
        # Get actual frame's ROI from memory map
        frame_path = self.of_result['frame_mmap_path']
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r',
                            shape=(self.sindy_result['n_frames'],
                                    self.sindy_result['roi_h'],
                                    self.sindy_result['roi_w']))
        bg_img = np.asarray(frame_mmap[idx])  # idx is 0-based

        extent = [0, u.shape[1]*meters_per_pixel, u.shape[0]*meters_per_pixel, 0]
        plt.figure(figsize=(8,6))
        plt.imshow(bg_img, cmap='gray', extent=extent, alpha=0.7)
        levels = np.linspace(np.nanmin(mag), np.nanmax(mag), 35)
        to_plot = np.nan_to_num(mag)
        plt.contourf(np.linspace(0,u.shape[1]*meters_per_pixel,u.shape[1]),
                     np.linspace(0,u.shape[0]*meters_per_pixel,u.shape[0]), to_plot,
                     levels=levels, cmap='jet', alpha=0.6, extend='both')
        plt.colorbar(label="Velocity magnitude")
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_stream_gui(self):
        nfr = self.sindy_result['n_frames']
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Frame", f"Frame index (1 - {nfr}):", 1, 1, nfr, 1)
        if not ok: return
        idx = idx-1
        which, ok = QtWidgets.QInputDialog.getItem(
            self, "Plot Field", "Display streamlines for:", ("Actual", "SINDy Prediction", "Error (diff)"), 0, False)
        if not ok: return
        meters_per_pixel = self.result['meters_per_pixel']
        roi_img = self.result['first_frame_roi']
        u_path, v_path = self.of_result['u_mmap_path'], self.of_result['v_mmap_path']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr,self.sindy_result['roi_h'],self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        u = np.asarray(u_mmap[idx])
        v = np.asarray(v_mmap[idx])
        u_pred = self.X_pred_optical[idx,:,:,0]
        v_pred = self.X_pred_optical[idx,:,:,1]
        if which=="Actual":
            plotu, plotv = u, v
            title = f"Frame {idx+1} Actual Velocity Streamlines"
        elif which=="SINDy Prediction":
            plotu, plotv = u_pred, v_pred
            title = f"Frame {idx+1} SINDy Prediction Streamlines"
        else:
            plotu, plotv = u-u_pred, v-v_pred
            title = f"Frame {idx+1} Error (Actual - SINDy) Streamlines"

        # Get actual frame's ROI from memory map
        frame_path = self.of_result['frame_mmap_path']
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r',
                            shape=(self.sindy_result['n_frames'],
                                    self.sindy_result['roi_h'],
                                    self.sindy_result['roi_w']))
        bg_img = np.asarray(frame_mmap[idx])  # idx is 0-based

        extent = [0, u.shape[1]*meters_per_pixel, u.shape[0]*meters_per_pixel, 0]
        x_vals = np.linspace(0,u.shape[1]*meters_per_pixel,u.shape[1])
        y_vals = np.linspace(0,u.shape[0]*meters_per_pixel,u.shape[0])
        Xg, Yg = np.meshgrid(x_vals, y_vals)
        plt.figure(figsize=(8,6))
        plt.imshow(bg_img, cmap='gray', extent=extent, alpha=0.7)
        plt.streamplot(Xg,Yg,plotu,plotv,color='w', density=1.2, linewidth=1, arrowsize=1.2)
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # --- EXPORTS/errors the same as before ---

    def show_sindy_equation_gui(self):
        sindy_model = self.sindy_result.get('model', None)
        if sindy_model is None:
            QtWidgets.QMessageBox.warning(self, "No SINDy Model", "Please run SINDy modeling first.")
            return
        eq_str = sindy_model.equations(precision=5)
        if isinstance(eq_str, list):
            eq_str = "\n\n".join(eq_str)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("SINDy Fitted Equation")
        dlg.resize(600, 400)
        vbox = QtWidgets.QVBoxLayout(dlg)
        text = QtWidgets.QPlainTextEdit()
        text.setPlainText(eq_str)
        text.setReadOnly(True)
        vbox.addWidget(text)
        copy_btn = QtWidgets.QPushButton("Copy to Clipboard")
        vbox.addWidget(copy_btn)
        def copy_to_clipboard():
            QtWidgets.QApplication.clipboard().setText(eq_str)
        copy_btn.clicked.connect(copy_to_clipboard)
        dlg.exec()

    def export_csv_gui(self):
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "./output"
        )
        if not output_dir:
            return  # User cancelled
        u_path = self.of_result['u_mmap_path']
        v_path = self.of_result['v_mmap_path']
        nfr = self.sindy_result['n_frames']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        self.status_bar.showMessage("Exporting CSV files...", 0)
        for frame_idx in range(u_mmap.shape[0]):
            frame_data = {
                'actual_u': u_mmap[frame_idx].flatten(),
                'actual_v': v_mmap[frame_idx].flatten(),
                'predicted_u': self.X_pred_optical[frame_idx, :, :, 0].flatten(),
                'predicted_v': self.X_pred_optical[frame_idx, :, :, 1].flatten()
            }
            df = pd.DataFrame(frame_data)
            df.to_csv(os.path.join(output_dir, f'frame_{frame_idx+1}_values.csv'), index=False)
            # Update status bar with progress
            self.status_bar.showMessage(f"Exporting CSV: Frame {frame_idx+1}/{u_mmap.shape[0]}", 0)
            QtWidgets.QApplication.processEvents()
        self.status_bar.showMessage(f'CSV files for frames saved to: {output_dir}', 5000)
        QtWidgets.QMessageBox.information(self, "Exported", f'CSV files for frames saved to:\n{output_dir}')

    def run_error_analysis_gui(self):
        u_path = self.of_result['u_mmap_path']
        v_path = self.of_result['v_mmap_path']
        nfr = self.sindy_result['n_frames']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr,self.sindy_result['roi_h'],self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        pred = self.X_pred_optical
        n_frames, h, w = u_mmap.shape
        rmse, mse = [], []
        for f in range(n_frames):
            err = (u_mmap[f] - pred[f,:,:,0])**2 + (v_mmap[f] - pred[f,:,:,1])**2
            mse.append(np.mean(err))
            rmse.append(np.sqrt(np.mean(err)))
        rmse, mse = np.array(rmse), np.array(mse)
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,8))
        ax1.plot(np.arange(1,n_frames+1),rmse)
        ax1.set_ylabel("RMSE")
        ax1.set_xlabel("Frame")
        ax1.set_title("RMSE per frame (Actual vs SINDy)")
        ax2.plot(np.arange(1,n_frames+1),mse)
        ax2.set_ylabel("MSE")
        ax2.set_xlabel("Frame")
        ax2.set_title("MSE per frame")
        plt.tight_layout()
        plt.show()
        idx, ok = QtWidgets.QInputDialog.getInt(self, "Frame for Spatial Error Map", f"Frame index (1-{n_frames})", 1, 1, n_frames, 1)
        if not ok:
            return
        idx -= 1
        err = np.sqrt((u_mmap[idx] - pred[idx,:,:,0])**2 + (v_mmap[idx] - pred[idx,:,:,1])**2)
        plt.figure(figsize=(8,6))
        plt.imshow(err, cmap='hot', origin='upper')
        plt.colorbar(label="Per-pixel RMSE")
        plt.title(f"Spatial Error Map: Frame {idx+1}")
        plt.tight_layout()
        plt.show()

    def export_animation_gui(self):
        if getattr(sys, 'frozen', False):  
            # Construct the path to the bundled ffmpeg.exe  
            # It will be in the 'ffmpeg_bin' folder relative to the executable's temporary path  
            bundled_ffmpeg_path = os.path.join(os.path.dirname(sys.argv[0]), "ffmpeg.exe")  

            # Set Matplotlib's animation.ffmpeg_path  
            plt.rcParams['animation.ffmpeg_path'] = bundled_ffmpeg_path  
            print(f"Matplotlib FFmpeg path set to: {bundled_ffmpeg_path}")  
        else:  
            # For development, Matplotlib will look for ffmpeg in your system's PATH  
            print("Running in development mode. Matplotlib will use system FFmpeg if available.")        
        u_path = self.of_result['u_mmap_path']
        v_path = self.of_result['v_mmap_path']
        nfr = self.sindy_result['n_frames']
        u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w']))
        v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
        pred = self.X_pred_optical
        n_frames, h, w = u_mmap.shape
        step = max(h // 30, 2)
        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        xg, yg = np.meshgrid(xs, ys)
        meters_per_pixel = self.result['meters_per_pixel']

        # --- Ask user for quiver scale and width ---
        scale, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver scale", "Set quiver scale (increase for shorter arrows):", 1, 0.1, 150, 2)
        if not ok: return
        width, ok = QtWidgets.QInputDialog.getDouble(
            self, "Quiver width", "Set quiver arrow width (e.g. 0.006):", 0.006, 0.0001, 0.1, 4)
        if not ok: return

        # Ask user for output file
        out_file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Animation",
            os.path.join(os.path.expanduser("~"), "quiver_animation.mp4"), "MP4 files (*.mp4);;All files (*)"
        )
        if not out_file:
            return
        self.status.setText("Rendering quiver animation. May take several minutes...")
        self.progress.show()
        self.status_bar.showMessage("Rendering quiver animation...", 0)
        fig, ax = plt.subplots(figsize=(8, 8))
        roi_img = self.result['first_frame_roi']
        # Update background image loading in animate function:
        frame_path = self.of_result['frame_mmap_path']
        frame_mmap = np.memmap(frame_path, dtype=np.uint8, mode='r',
                            shape=(self.sindy_result['n_frames'],
                                    self.sindy_result['roi_h'],
                                    self.sindy_result['roi_w']))
        extent = [0, w * meters_per_pixel, h * meters_per_pixel, 0]

        def animate(fidx):
            bg_img = frame_mmap[fidx]
            ax.clear()
            ax.imshow(bg_img, cmap='gray', origin='upper', extent=extent)
            # Actual quiver (blue)
            ax.quiver(
                xg * meters_per_pixel, yg * meters_per_pixel,
                u_mmap[fidx][ys[:, None], xs[None, :]],
                v_mmap[fidx][ys[:, None], xs[None, :]],
                color='b', angles='xy', scale=scale, width=width, label='Actual'
            )
            # SINDy Prediction quiver (red)
            ax.quiver(
                xg * meters_per_pixel, yg * meters_per_pixel,
                pred[fidx, ys[:, None], xs[None, :], 0],
                pred[fidx, ys[:, None], xs[None, :], 1],
                color='r', angles='xy', scale=scale, width=width, label='SINDy'
            )
            ax.set_title(f"Frame {fidx + 1}: Quiver Overlay (Blue=Actual, Red=SINDy)")
            ax.set_xlabel('x (meters)')
            ax.set_ylabel('y (meters)')
            ax.legend()
            self.status_bar.showMessage(f"Rendering quiver animation: Frame {fidx + 1}/{n_frames}", 0)
            QtWidgets.QApplication.processEvents()

        ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
        from matplotlib.animation import FFMpegWriter
        ani.save(out_file, writer=FFMpegWriter(fps=8))
        plt.close(fig)
        self.status.setText(f"Animation export complete: {out_file}")
        self.status_bar.showMessage(f"Animation export complete: {out_file}", 5000)
        QtWidgets.QMessageBox.information(self, "Done", f"Animation saved:\n{out_file}")
        self.progress.hide()

    # def export_streamline_animation_gui(self):
    #    u_path = self.of_result['u_mmap_path']
    #     v_path = self.of_result['v_mmap_path']
    #     nfr = self.sindy_result['n_frames']
    #     u_mmap = np.memmap(u_path, np.float32, mode='r', shape=(nfr, self.sindy_result['roi_h'], self.sindy_result['roi_w']))
    #     v_mmap = np.memmap(v_path, np.float32, mode='r', shape=u_mmap.shape)
    #     pred = self.X_pred_optical
    #     n_frames, h, w = u_mmap.shape
    #     meters_per_pixel = self.result['meters_per_pixel']
    #     img = self.result['first_frame_roi']
    #     bg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    #     extent = [0, w*meters_per_pixel, h*meters_per_pixel, 0]
    #     # Ask user for output file
    #     out_file, _ = QtWidgets.QFileDialog.getSaveFileName(
    #         self, "Save Animation", os.path.join(os.path.expanduser("~"), "streamline_animation.mp4"), "MP4 files (*.mp4);;All files (*)"
    #     )
    #     if not out_file:
    #         return
    #     self.status.setText("Rendering streamline animation...")
    #     self.progress.show()
    #     self.status_bar.showMessage("Rendering streamline animation...", 0)
    #     fig, axs = plt.subplots(1, 2, figsize=(14,6))
    #     X = np.linspace(0, w*meters_per_pixel, w)
    #     Y = np.linspace(0, h*meters_per_pixel, h)
    #     Xg, Yg = np.meshgrid(X, Y)
    #     def animate(fidx):
    #         for ax in axs: ax.clear()
    #         axs[0].imshow(bg, cmap='gray', extent=extent, alpha=0.5)
    #         axs[1].imshow(bg, cmap='gray', extent=extent, alpha=0.5)
    #         axs[0].streamplot(Xg,Yg, u_mmap[fidx], v_mmap[fidx], color='b',density=1.3,arrowsize=1.3)
    #         axs[1].streamplot(Xg,Yg,pred[fidx,:,:,0], pred[fidx,:,:,1], color='r',density=1.3,arrowsize=1.3)
    #         axs[0].set_title(f'Actual - Frame {fidx+1}')
    #         axs[1].set_title(f'SINDy Predicted - Frame {fidx+1}')
    #         for ax in axs:
    #             ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    #         # Update status bar with progress
    #         self.status_bar.showMessage(f"Rendering streamline animation: Frame {fidx+1}/{n_frames}", 0)
    #         QtWidgets.QApplication.processEvents()
    #     ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
    #     from matplotlib.animation import FFMpegWriter
    #     ani.save(out_file, writer=FFMpegWriter(fps=8))
    #     plt.close(fig)
    #     self.status.setText(f"Streamline Animation saved: {out_file}")
    #     self.status_bar.showMessage(f"Streamline Animation saved: {out_file}", 5000)
    #     QtWidgets.QMessageBox.information(self, "Done", f"Streamline animation saved:\n{out_file}")
    #     self.progress.hide() */

import sys
import os
import time # Make sure time is imported if you're using time.sleep
from PyQt6 import QtWidgets
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter # QPainter is needed for dummy pixmap
from PyQt6.QtCore import Qt # Qt.AlignmentFlag, Qt.GlobalColor

# Assuming FluidGui is defined elsewhere in your code
# from your_module import FluidGui

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Determine the base path for resources
    # When bundled by Nuitka (or PyInstaller), sys.argv[0] points to the executable.
    # os.path.dirname(sys.argv[0]) gives the directory where the executable is located
    # (which is the temporary extraction directory for --onefile builds).
    # This is the most reliable way to find bundled data files.
    application_base_path = os.path.dirname(sys.argv[0])

    # Construct the full path to the logo.png
    # Assuming logo.png is bundled at the root of the extracted directory
    # using --include-data-file="logo.png=logo.png"
    logo_full_path = os.path.join(application_base_path, "logo.png")

    # Create a dummy splash_image.png if it doesn't exist for testing
    # This logic will now check the *expected bundled path* first.
    # If running unbundled, application_base_path will be your script's directory.
    if not os.path.exists(logo_full_path):
        print(f"Warning: Logo file not found at '{logo_full_path}'. Attempting to create a dummy logo.")
        try:
            dummy_pixmap = QPixmap(400, 200)
            dummy_pixmap.fill(QColor("lightblue"))
            painter = QPainter(dummy_pixmap)
            painter.setFont(QFont("Arial", 24))
            painter.drawText(dummy_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "App Loading...")
            painter.end()

            # Save the dummy logo to the expected path
            dummy_pixmap.save(logo_full_path)
            print(f"Created dummy logo at '{logo_full_path}'")
        except Exception as e:
            print(f"FATAL: Could not create dummy logo at '{logo_full_path}': {e}")
            # If dummy creation fails, you might want to exit or show a simple message box
            sys.exit(1) # Exit if we can't even get a dummy logo

    # Create and show the splash screen
    # Use the full path to load the pixmap
    splash_pix = QPixmap(logo_full_path)

    # Check if pixmap loaded successfully
    if splash_pix.isNull():
        print(f"FATAL: Failed to load QPixmap from '{logo_full_path}'. Exiting.")
        sys.exit(1)

    splash = QtWidgets.QSplashScreen(splash_pix)
    splash.setFont(QFont("Arial", 12))
    splash.showMessage("Initializing application...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, QColor("blue"))
    splash.show()
    QtWidgets.QApplication.processEvents() # Ensure splash screen is painted

    # Simulate loading time for modules and app initialization
    loading_steps = [
        "Loading core modules...",
        "Setting up UI...",
        "Preparing data structures...",
        "Finalizing startup..."
    ]
    for i, step in enumerate(loading_steps):
        splash.showMessage(f"{step} ({i+1}/{len(loading_steps)})", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, QColor("blue"))
        QtWidgets.QApplication.processEvents() # Update splash screen
        time.sleep(0.5) # Simulate work

    win = FluidGui() # Assuming FluidGui is your main application window class
    win.show()

    splash.finish(win) # Close the splash screen when the main window is ready

    sys.exit(app.exec())
if __name__ == "__main__":
    main()