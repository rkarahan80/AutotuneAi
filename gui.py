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
        
        # Left panel - Basic controls
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
        controls_layout.addWidget(basic_controls)
        
        # Right panel - Effect racks and new controls
        effects_and_new_controls_panel = QVBoxLayout() # Renamed for clarity
        
        # Effect racks (Original)
        self.delay_rack = EffectRack("Delay") # Store as instance var if needed later
        self.flanger_rack = EffectRack("Flanger")
        # filter_rack = EffectRack("Filter") # This will be replaced by Parametric EQ
        
        effects_and_new_controls_panel.addWidget(self.delay_rack)
        effects_and_new_controls_panel.addWidget(self.flanger_rack)
        # effects_and_new_controls_panel.addWidget(filter_rack) # Removed

        # Reverb Controls GroupBox
        reverb_group = QGroupBox("Reverb")
        reverb_layout = QVBoxLayout()

        # Reverb Decay Time
        decay_layout = QHBoxLayout()
        decay_label = QLabel("Decay Time (s):")
        self.reverb_decay_spin = QDoubleSpinBox()
        self.reverb_decay_spin.setRange(0.0, 5.0)
        self.reverb_decay_spin.setValue(0.0) # Default off
        self.reverb_decay_spin.setSingleStep(0.1)
        decay_layout.addWidget(decay_label)
        decay_layout.addWidget(self.reverb_decay_spin)
        reverb_layout.addLayout(decay_layout)

        # Reverb Damping
        damping_layout = QHBoxLayout()
        damping_label = QLabel("Damping (0-1):")
        self.reverb_damping_spin = QDoubleSpinBox()
        self.reverb_damping_spin.setRange(0.0, 1.0)
        self.reverb_damping_spin.setValue(0.5)
        self.reverb_damping_spin.setSingleStep(0.1)
        damping_layout.addWidget(damping_label)
        damping_layout.addWidget(self.reverb_damping_spin)
        reverb_layout.addLayout(damping_layout)

        # Reverb Mix
        mix_layout = QHBoxLayout()
        mix_label = QLabel("Wet/Dry Mix (0-1):")
        self.reverb_mix_spin = QDoubleSpinBox()
        self.reverb_mix_spin.setRange(0.0, 1.0)
        self.reverb_mix_spin.setValue(0.0) # Default off
        self.reverb_mix_spin.setSingleStep(0.05)
        mix_layout.addWidget(mix_label)
        mix_layout.addWidget(self.reverb_mix_spin)
        reverb_layout.addLayout(mix_layout)

        reverb_group.setLayout(reverb_layout)
        effects_and_new_controls_panel.addWidget(reverb_group)

        # Parametric EQ Controls GroupBox
        eq_group = QGroupBox("Parametric EQ")
        eq_layout = QVBoxLayout()

        # EQ Filter Type
        type_layout = QHBoxLayout()
        type_label = QLabel("Filter Type:")
        self.eq_type_combo = QComboBox()
        self.eq_type_combo.addItems(["Off", "Lowpass", "Highpass", "Bandpass", "Bandstop"])
        self.eq_type_combo.setCurrentText("Off")
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.eq_type_combo)
        eq_layout.addLayout(type_layout)

        # EQ Center Frequency
        freq_layout = QHBoxLayout()
        freq_label = QLabel("Center/Cutoff Freq (Hz):")
        self.eq_freq_spin = QDoubleSpinBox()
        self.eq_freq_spin.setRange(20.0, 20000.0)
        self.eq_freq_spin.setValue(1000.0)
        self.eq_freq_spin.setSingleStep(10.0) # Could be more dynamic
        # self.eq_freq_spin.setDecimals(0) # For cleaner display of common Hz values
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.eq_freq_spin)
        eq_layout.addLayout(freq_layout)

        # EQ Q Factor
        q_layout = QHBoxLayout()
        q_label = QLabel("Q Factor:")
        self.eq_q_spin = QDoubleSpinBox()
        self.eq_q_spin.setRange(0.1, 10.0)
        self.eq_q_spin.setValue(1.0)
        self.eq_q_spin.setSingleStep(0.1)
        q_layout.addWidget(q_label)
        q_layout.addWidget(self.eq_q_spin)
        eq_layout.addLayout(q_layout)

        # EQ Order
        order_layout = QHBoxLayout()
        order_label = QLabel("Order:")
        self.eq_order_spin = QSpinBox()
        self.eq_order_spin.setRange(1, 8) # Butterworth typically uses even orders, but allow odd.
        self.eq_order_spin.setValue(4)
        order_layout.addWidget(order_label)
        order_layout.addWidget(self.eq_order_spin)
        eq_layout.addLayout(order_layout)

        eq_group.setLayout(eq_layout)
        effects_and_new_controls_panel.addWidget(eq_group)
        
        controls_layout.addLayout(effects_and_new_controls_panel) # Add the QVBoxLayout to the main QHBoxLayout
        
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
            'add_effects': self.effects_check.isChecked(), # Controls delay/flanger from main.py
            'beat_slice': self.beat_slice_check.isChecked(),
            'add_sidechain': self.sidechain_check.isChecked(),

            # Reverb parameters
            'reverb_decay_time': self.reverb_decay_spin.value(),
            'reverb_damping': self.reverb_damping_spin.value(),
            'reverb_mix': self.reverb_mix_spin.value(),

            # EQ parameters
            'eq_filter_type': self.eq_type_combo.currentText().lower() if self.eq_type_combo.currentText() != "Off" else None,
            'eq_center_freq': self.eq_freq_spin.value(),
            'eq_q_factor': self.eq_q_spin.value(),
            'eq_order': self.eq_order_spin.value()
        }

        # Debug print for params
        print("Remix Parameters:", params)

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