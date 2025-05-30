"""
Premiermixx GUI - Yapay Zeka Destekli Müzik Remix Sistemi
Copyright (c) 2024 Premiermixx
MIT License - Detaylar için LICENSE dosyasına bakın
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                            QSlider, QCheckBox, QProgressBar, QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QFont
import os
from main import Premiermixx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class RemixWorker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, input_file, output_file, params):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.params = params

    def run(self):
        try:
            remixer = Premiermixx(self.input_file, self.output_file)
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

    def initUI(self):
        self.setWindowTitle('Premiermixx - AI Remix Studio')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #357abd;
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
            }
        """)

        # Ana widget ve layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Dosya seçim bölümü
        file_layout = QHBoxLayout()
        self.file_label = QLabel('Seçili Dosya: Yok')
        select_button = QPushButton('Dosya Seç')
        select_button.clicked.connect(self.selectFile)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(select_button)
        layout.addLayout(file_layout)

        # Parametre kontrolleri
        params_layout = QVBoxLayout()

        # Tempo kontrolü
        tempo_layout = QHBoxLayout()
        tempo_label = QLabel('Tempo Değişimi:')
        self.tempo_spin = QDoubleSpinBox()
        self.tempo_spin.setRange(0.5, 2.0)
        self.tempo_spin.setValue(1.0)
        self.tempo_spin.setSingleStep(0.1)
        tempo_layout.addWidget(tempo_label)
        tempo_layout.addWidget(self.tempo_spin)
        params_layout.addLayout(tempo_layout)

        # Pitch kontrolü
        pitch_layout = QHBoxLayout()
        pitch_label = QLabel('Perde Kaydırma:')
        self.pitch_spin = QSpinBox()
        self.pitch_spin.setRange(-12, 12)
        self.pitch_spin.setValue(0)
        pitch_layout.addWidget(pitch_label)
        pitch_layout.addWidget(self.pitch_spin)
        params_layout.addLayout(pitch_layout)

        # Efekt kontrolleri
        self.effects_check = QCheckBox('Efektleri Etkinleştir')
        self.effects_check.setChecked(True)
        params_layout.addWidget(self.effects_check)

        self.beat_slice_check = QCheckBox('Beat Slicing')
        self.beat_slice_check.setChecked(True)
        params_layout.addWidget(self.beat_slice_check)

        self.sidechain_check = QCheckBox('Sidechain Kompresyon')
        self.sidechain_check.setChecked(True)
        params_layout.addWidget(self.sidechain_check)

        layout.addLayout(params_layout)

        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Remix butonu
        self.remix_button = QPushButton('Remix Başlat')
        self.remix_button.clicked.connect(self.startRemix)
        self.remix_button.setEnabled(False)
        layout.addWidget(self.remix_button)

        # Görselleştirme alanı
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def selectFile(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Ses Dosyası Seç', '', 'Ses Dosyaları (*.wav *.mp3 *.ogg)')
        if file_name:
            self.input_file = file_name
            self.file_label.setText(f'Seçili Dosya: {os.path.basename(file_name)}')
            self.remix_button.setEnabled(True)

    def startRemix(self):
        if not self.input_file:
            return

        params = {
            'tempo_change': self.tempo_spin.value(),
            'pitch_steps': self.pitch_spin.value(),
            'add_effects': self.effects_check.isChecked(),
            'beat_slice': self.beat_slice_check.isChecked(),
            'add_sidechain': self.sidechain_check.isChecked()
        }

        self.worker = RemixWorker(self.input_file, self.output_file, params)
        self.worker.finished.connect(self.onRemixFinished)
        self.worker.error.connect(self.onRemixError)
        
        self.remix_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # Belirsiz ilerleme
        self.worker.start()

    def onRemixFinished(self):
        self.remix_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        # Görselleştirmeyi güncelle
        self.updateVisualization()

    def onRemixError(self, error_msg):
        self.remix_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        # Hata mesajını göster

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