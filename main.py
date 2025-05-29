"""
Premiermixx - Yapay Zeka Destekli Müzik Remix Sistemi
Copyright (c) 2024 Premiermixx
MIT License - Detaylar için LICENSE dosyasına bakın
"""

import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

class Premiermixx:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.audio = None
        self.sr = None
        self.processed_audio = None
        
    def load_audio(self):
        """Ses dosyasını yükle"""
        self.audio, self.sr = librosa.load(self.input_file, sr=None)
        
    def apply_beat_detection(self):
        """Ritim tespiti ve senkronizasyonu"""
        tempo, beat_frames = librosa.beat.beat_track(y=self.audio, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        return tempo, beat_times
    
    def time_stretch(self, rate=1.0):
        """Zaman uzatma/kısaltma efekti"""
        self.audio = librosa.effects.time_stretch(self.audio, rate=rate)
        
    def pitch_shift(self, steps):
        """Perde kaydırma"""
        self.audio = librosa.effects.pitch_shift(self.audio, sr=self.sr, n_steps=steps)
        
    def add_delay(self, delay_time=0.3, decay=0.5):
        """Delay efekti ekle"""
        delay_samples = int(self.sr * delay_time)
        delay = np.zeros_like(self.audio)
        delay[delay_samples:] = self.audio[:-delay_samples] * decay
        self.audio = self.audio + delay
        
    def add_filter(self, filter_type='lowpass', cutoff_freq=1000):
        """Filtre uygula"""
        if filter_type == 'lowpass':
            self.audio = librosa.effects.preemphasis(self.audio, coef=cutoff_freq/self.sr)
        elif filter_type == 'highpass':
            self.audio = librosa.effects.preemphasis(self.audio, coef=-cutoff_freq/self.sr)
            
    def add_flanger(self, rate=0.5, depth=0.002):
        """Flanger efekti"""
        t = np.arange(len(self.audio)) / self.sr
        mod = np.sin(2 * np.pi * rate * t)
        delay_samples = int(depth * self.sr)
        flanger = np.zeros_like(self.audio)
        for i in range(len(self.audio)):
            delay = int(delay_samples * (1 + mod[i]))
            if i >= delay:
                flanger[i] = self.audio[i - delay]
        self.audio = 0.7 * self.audio + 0.3 * flanger
        
    def create_loop(self, start_time, end_time, repeats=4):
        """Belirli bir bölümü döngüye al"""
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        loop_section = self.audio[start_sample:end_sample]
        loop = np.tile(loop_section, repeats)
        return loop
    
    def visualize_remix(self, output_file='remix_analysis.png'):
        """Remix görselleştirmesi"""
        plt.figure(figsize=(15, 8))
        
        # Dalga formu
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, len(self.audio)/self.sr, len(self.audio)), self.audio)
        plt.title('Remix Dalga Formu')
        plt.xlabel('Zaman (saniye)')
        plt.ylabel('Genlik')
        
        # Spektrogram
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio)), ref=np.max)
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Remix Spektrogramı')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    def save_remix(self):
        """Remixlenmiş sesi kaydet"""
        sf.write(self.output_file, self.audio, self.sr)
        
    def process_remix(self, tempo_change=1.0, pitch_steps=0, add_effects=True):
        """Remix işlem pipeline'ı"""
        print("Ses dosyası yükleniyor...")
        self.load_audio()
        
        print("Tempo ve ritim analizi yapılıyor...")
        original_tempo, beats = self.apply_beat_detection()
        
        if tempo_change != 1.0:
            print(f"Tempo değiştiriliyor ({tempo_change}x)...")
            self.time_stretch(tempo_change)
            
        if pitch_steps != 0:
            print(f"Perde kaydırma uygulanıyor ({pitch_steps} adım)...")
            self.pitch_shift(pitch_steps)
            
        if add_effects:
            print("Efektler ekleniyor...")
            self.add_delay(0.3, 0.4)
            self.add_flanger(0.7, 0.003)
            self.add_filter('lowpass', 2000)
        
        print("Remix görselleştirmesi oluşturuluyor...")
        self.visualize_remix()
        
        print("Remix kaydediliyor...")
        self.save_remix()
        print(f"Remix tamamlandı! Çıktı: {self.output_file}")

def main():
    print("""
    ╔═══════════════════════════════════════╗
    ║            Premiermixx                ║
    ║     Yapay Zeka Destekli Remix        ║
    ╚═══════════════════════════════════════╝
    """)
    
    input_file = "input.wav"
    output_file = "output_remix.wav"
    
    try:
        remixer = Premiermixx(input_file, output_file)
        remixer.process_remix(
            tempo_change=1.2,      # 20% daha hızlı
            pitch_steps=2,         # 2 adım yukarı
            add_effects=True       # Efektleri ekle
        )
        
    except FileNotFoundError:
        print("Lütfen 'input.wav' adında bir ses dosyası ekleyin.")
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()