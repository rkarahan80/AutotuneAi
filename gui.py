"""
Premiermixx GUI - Yapay Zeka Destekli Müzik Remix Sistemi
Copyright (c) 2024 Premiermixx
MIT License - Detaylar için LICENSE dosyasına bakın
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                            QSlider, QCheckBox, QProgressBar, QSpinBox, QDoubleSpinBox,
                            QComboBox, QTabWidget, QGroupBox, QDial, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette, QLinearGradient
import os
from main import Premiermixx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.waveform_data = None
        
    def update_waveform(self, audio_data):
        self.waveform_data = audio_data
        self.update()
        
    def paintEvent(self, event):
        if self.waveform_data is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            width = self.width()
            height = self.height()
            center_y = height // 2
            
            # Draw background
            gradient = QLinearGradient(0, 0, 0, height)
            gradient.setColorAt(0, QColor(40, 40, 40))
            gradient.setColorAt(1, QColor(30, 30, 30))
            painter.fillRect(0, 0, width, height, gradient)
            
            # Draw waveform
            painter.setPen(QPen(QColor(74, 144, 226), 1))
            points_per_pixel = len(self.waveform_data) // width
            for x in range(width):
                start_idx = x * points_per_pixel
                end_idx = start_idx + points_per_pixel
                if start_idx < len(self.waveform_data):
                    chunk = self.waveform_data[start_idx:end_idx]
                    if len(chunk) > 0:
                        amplitude = np.max(np.abs(chunk)) * height / 2
                        painter.drawLine(x, center_y - amplitude, x, center_y + amplitude)

class EffectRack(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4a4a4a;
                border-radius: 5px;
                margin-top: 1em;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #ffffff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Effect controls
        self.amount_dial = QDial()
        self.amount_dial.setRange(0, 100)
        self.amount_dial.setValue(50)
        self.amount_dial.setNotchesVisible(True)
        
        self.mix_slider = QSlider(Qt.Orientation.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(100)
        
        # Labels
        amount_label = QLabel("Amount")
        amount_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mix_label = QLabel("Mix")
        mix_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(amount_label)
        layout.addWidget(self.amount_dial)
        layout.addWidget(mix_label)
        layout.addWidget(self.mix_slider)

class RemixWorker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    waveform_update = pyqtSignal(np.ndarray)

    def __init__(self, input_file, output_file, params):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.params = params

    def run(self):
        try:
            remixer = Premiermixx(self.input_file, self.output_file)
            
            # Emit waveform data for visualization
            self.waveform_update.emit(remixer.audio)
            
            # Process with progress updates
            total_steps = 5
            for i in range(total_steps):
                self.progress.emit(int((i + 1) * 100 / total_steps))
                # Simulate processing steps
                QThread.msleep(500)
            
            remixer.process_remix(**self.params)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class PremiermixxGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file = ""
        self.output_file = "output_remix.wav"
        self.initUI()
        
        # Setup timer for VU meter animation
        self.vu_timer = QTimer()
        self.vu_timer.timeout.connect(self.updateVUMeter)
        self.vu_timer.start(50)
        
        self.vu_level = 0

    def initUI(self):
        self.setWindowTitle('Premiermixx - AI Remix Studio')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2868a9;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #4a4a4a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QCheckBox {
                color: white;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #4a4a4a;
                background: #2b2b2b;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #4a90e2;
                background: #4a90e2;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #4a4a4a;
                background: #2b2b2b;
            }
            QTabBar::tab {
                background: #1e1e1e;
                color: white;
                padding: 8px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #4a90e2;
            }
            QComboBox {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #4a4a4a;
                padding: 5px;
                border-radius: 3px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #4a4a4a;
                padding: 5px;
                border-radius: 3px;
            }
            QDial {
                background-color: #4a90e2;
            }
        """)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header section
        header_layout = QHBoxLayout()
        logo_label = QLabel('PREMIERMIXX')
        logo_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #4a90e2;
        """)
        header_layout.addWidget(logo_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # Tab widget
        tab_widget = QTabWidget()
        
        # Main tab
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout(main_tab)
        
        # File selection section
        file_frame = QFrame()
        file_frame.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; padding: 10px;")
        file_layout = QHBoxLayout(file_frame)
        self.file_label = QLabel('No file selected')
        select_button = QPushButton('Select File')
        select_button.clicked.connect(self.selectFile)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(select_button)
        main_tab_layout.addWidget(file_frame)

        # Waveform display
        self.waveform_widget = WaveformWidget()
        main_tab_layout.addWidget(self.waveform_widget)

        # Controls section
        controls_layout = QHBoxLayout()
        
        # Left panel - Basic controls and Autotune
        left_panel_layout = QVBoxLayout()

        basic_controls = QGroupBox("Basic Controls")
        basic_layout = QVBoxLayout()
        
        # Tempo control
        tempo_layout = QHBoxLayout()
        tempo_label = QLabel('Tempo:')
        self.tempo_spin = QDoubleSpinBox()
        self.tempo_spin.setRange(0.5, 2.0)
        self.tempo_spin.setValue(1.0)
        self.tempo_spin.setSingleStep(0.1)
        tempo_layout.addWidget(tempo_label)
        tempo_layout.addWidget(self.tempo_spin)
        basic_layout.addLayout(tempo_layout)
        
        # Pitch control
        pitch_layout = QHBoxLayout()
        pitch_label = QLabel('Pitch:')
        self.pitch_spin = QSpinBox()
        self.pitch_spin.setRange(-12, 12)
        self.pitch_spin.setValue(0)
        pitch_layout.addWidget(pitch_label)
        pitch_layout.addWidget(self.pitch_spin)
        basic_layout.addLayout(pitch_layout)
        basic_controls.setLayout(basic_layout)
        left_panel_layout.addWidget(basic_controls)

        # Autotune controls
        autotune_controls = QGroupBox("AI Autotune")
        autotune_layout = QVBoxLayout()

        self.autotune_check = QCheckBox("Enable Autotune")
        autotune_layout.addWidget(self.autotune_check)

        # Autotune Strength
        autotune_strength_layout = QHBoxLayout()
        autotune_strength_label = QLabel("Strength:")
        self.autotune_strength_spin = QDoubleSpinBox()
        self.autotune_strength_spin.setRange(0.0, 1.0)
        self.autotune_strength_spin.setValue(0.8)
        self.autotune_strength_spin.setSingleStep(0.1)
        autotune_strength_layout.addWidget(autotune_strength_label)
        autotune_strength_layout.addWidget(self.autotune_strength_spin)
        autotune_layout.addLayout(autotune_strength_layout)

        # Autotune Model
        autotune_model_layout = QHBoxLayout()
        autotune_model_label = QLabel("Model:")
        self.autotune_model_combo = QComboBox()
        self.autotune_model_combo.addItems(["tiny", "full"])
        autotune_model_layout.addWidget(autotune_model_label)
        autotune_model_layout.addWidget(self.autotune_model_combo)
        autotune_layout.addLayout(autotune_model_layout)

        # Autotune Confidence Threshold
        autotune_confidence_layout = QHBoxLayout()
        autotune_confidence_label = QLabel("Confidence:")
        self.autotune_confidence_spin = QDoubleSpinBox()
        self.autotune_confidence_spin.setRange(0.01, 0.99) # Crepe confidence typically > 0
        self.autotune_confidence_spin.setValue(0.4)
        self.autotune_confidence_spin.setSingleStep(0.05)
        autotune_confidence_layout.addWidget(autotune_confidence_label)
        autotune_confidence_layout.addWidget(self.autotune_confidence_spin)
        autotune_layout.addLayout(autotune_confidence_layout)

        autotune_controls.setLayout(autotune_layout)
        left_panel_layout.addWidget(autotune_controls)

        # AI Source Separation controls
        separation_controls = QGroupBox("AI Source Separation (Spleeter)")
        separation_layout = QVBoxLayout()

        self.separation_check = QCheckBox("Enable Separation")
        separation_layout.addWidget(self.separation_check)

        separation_model_layout = QHBoxLayout()
        separation_model_label = QLabel("Model:")
        self.separation_model_combo = QComboBox()
        self.separation_model_combo.addItems([
            "spleeter:2stems",
            "spleeter:4stems",
            "spleeter:5stems"
        ])
        separation_model_layout.addWidget(separation_model_label)
        separation_model_layout.addWidget(self.separation_model_combo)
        separation_layout.addLayout(separation_model_layout)

        # Output directory for stems - for now, a fixed default is used in main.py
        # Later, can add QLineEdit and QFileDialog to select this
        # self.stems_output_dir_label = QLabel("Output Directory: (default used)")
        # separation_layout.addWidget(self.stems_output_dir_label)

        separation_controls.setLayout(separation_layout)
        left_panel_layout.addWidget(separation_controls)

        left_panel_layout.addStretch()

        controls_layout.addLayout(left_panel_layout)
        
        # Right panel - Effect racks
        effects_panel = QVBoxLayout()
        
        # Effect racks
        delay_rack = EffectRack("Delay")
        flanger_rack = EffectRack("Flanger")
        filter_rack = EffectRack("Filter")
        
        effects_panel.addWidget(delay_rack)
        effects_panel.addWidget(flanger_rack)
        effects_panel.addWidget(filter_rack)
        
        controls_layout.addLayout(effects_panel)
        
        main_tab_layout.addLayout(controls_layout)
        
        # Checkboxes for additional features
        features_layout = QHBoxLayout()
        self.effects_check = QCheckBox('Enable Effects')
        self.beat_slice_check = QCheckBox('Beat Slicing')
        self.sidechain_check = QCheckBox('Sidechain')
        features_layout.addWidget(self.effects_check)
        features_layout.addWidget(self.beat_slice_check)
        features_layout.addWidget(self.sidechain_check)
        main_tab_layout.addLayout(features_layout)

        # Progress section
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4a4a4a;
                border-radius: 5px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
                border-radius: 3px;
            }
        """)
        main_tab_layout.addWidget(self.progress_bar)

        # Process button
        self.remix_button = QPushButton('Start Remix')
        self.remix_button.clicked.connect(self.startRemix)
        self.remix_button.setEnabled(False)
        main_tab_layout.addWidget(self.remix_button)

        # Add main tab to tab widget
        tab_widget.addTab(main_tab, "Main")
        
        # Add visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        tab_widget.addTab(viz_tab, "Analysis")

        main_layout.addWidget(tab_widget)

    def updateVUMeter(self):
        # Simulate VU meter movement
        self.vu_level = min(100, max(0, self.vu_level + np.random.normal(0, 10)))
        self.progress_bar.setValue(int(self.vu_level))

    def selectFile(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Select Audio File', '', 'Audio Files (*.wav *.mp3 *.ogg)')
        if file_name:
            self.input_file = file_name
            self.file_label.setText(f'Selected: {os.path.basename(file_name)}')
            self.remix_button.setEnabled(True)

    def startRemix(self):
        if not self.input_file:
            return

        params = {
            'tempo_change': self.tempo_spin.value(),
            'pitch_steps': self.pitch_spin.value(),
            'add_effects': self.effects_check.isChecked(),
            'beat_slice': self.beat_slice_check.isChecked(),
            'add_sidechain': self.sidechain_check.isChecked(),
            'apply_autotune': self.autotune_check.isChecked(),
            'autotune_strength': self.autotune_strength_spin.value(),
            'autotune_model': self.autotune_model_combo.currentText(),
            'autotune_confidence': self.autotune_confidence_spin.value(),
            'apply_source_separation': self.separation_check.isChecked(),
            'spleeter_model': self.separation_model_combo.currentText(),
            'stems_output_dir': 'gui_separated_stems' # Default output dir for GUI initiated separations
        }

        self.worker = RemixWorker(self.input_file, self.output_file, params)
        self.worker.finished.connect(self.onRemixFinished)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self.onRemixError)
        self.worker.waveform_update.connect(self.waveform_widget.update_waveform)
        
        self.remix_button.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.worker.start()

    def onRemixFinished(self):
        self.remix_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.updateVisualization()

    def onRemixError(self, error_msg):
        self.remix_button.setEnabled(True)
        self.progress_bar.setValue(0)
        # Show error message to user

    def updateVisualization(self):
        if os.path.exists('remix_analysis.png'):
            self.ax.clear()
            img = plt.imread('remix_analysis.png')
            self.ax.imshow(img)
            self.ax.axis('off')
            self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    ex = PremiermixxGUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()