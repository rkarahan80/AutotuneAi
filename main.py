"""
AutotuneAI - Yapay Zeka Destekli Ses İşleme ve Autotune Sistemi
Copyright (c) 2024 AutotuneAI
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

class AutotuneAI:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.audio = None
        self.sr = None
        self.corrected_audio = None
        
    def load_audio(self):
        """Ses dosyasını yükle"""
        self.audio, self.sr = librosa.load(self.input_file, sr=None)
        
    def pitch_correction(self, threshold=0.3):
        """Pitch düzeltme işlemi"""
        # Pitch tespiti
        pitches, magnitudes = librosa.piptrack(y=self.audio, sr=self.sr)
        
        # En belirgin pitch'i seç
        pitch_median = medfilt(np.mean(pitches, axis=0), kernel_size=11)
        
        # WORLD vocoderi ile ses işleme
        _f0, t = pw.dio(self.audio.astype(np.float64), self.sr)
        f0 = pw.stonemask(self.audio.astype(np.float64), _f0, t, self.sr)
        sp = pw.cheaptrick(self.audio.astype(np.float64), f0, t, self.sr)
        ap = pw.d4c(self.audio.astype(np.float64), f0, t, self.sr)
        
        # Pitch düzeltme
        f0_corrected = np.copy(f0)
        for i in range(len(f0)):
            if f0[i] > 0:
                f0_corrected[i] = self.adjust_to_nearest_note(f0[i])
        
        self.corrected_audio = pw.synthesize(f0_corrected, sp, ap, self.sr)
        
    def adjust_to_nearest_note(self, freq):
        """En yakın müzikal notaya ayarlama"""
        A4 = 440.0
        notes = 12 * np.log2(freq / A4)
        notes = np.round(notes)
        return A4 * 2.0 ** (notes / 12.0)
    
    def add_echo(self, delay=0.3, decay=0.5):
        """Ses kaydına eko efekti ekle"""
        delay_samples = int(self.sr * delay)
        echo = np.zeros_like(self.audio)
        echo[delay_samples:] = self.audio[:-delay_samples] * decay
        self.audio = self.audio + echo
        
    def change_speed(self, speed_factor):
        """Ses hızını değiştir"""
        self.audio = librosa.effects.time_stretch(self.audio, rate=speed_factor)
        
    def add_reverb(self, reverb_strength=0.5):
        """Reverb efekti ekle"""
        reverb = np.zeros_like(self.audio)
        delay_samples = int(self.sr * 0.05)  # 50ms delay
        for i in range(5):  # 5 yankı
            delay = delay_samples * (i + 1)
            if delay < len(self.audio):
                reverb[delay:] += self.audio[:-delay] * (reverb_strength ** (i + 1))
        self.audio = self.audio + reverb
        
    def analyze_audio(self):
        """Ses analizi yap ve sonuçları döndür"""
        # Temel frekans analizi
        f0, voiced_flag, voiced_probs = librosa.pyin(self.audio, fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'))
        
        # Ses yüksekliği analizi
        rms = librosa.feature.rms(y=self.audio)[0]
        
        # Spektral merkezoid
        spectral_centroids = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr)[0]
        
        return {
            'ortalama_frekans': np.mean(f0[voiced_flag]),
            'max_ses_seviyesi': np.max(rms),
            'ortalama_ses_seviyesi': np.mean(rms),
            'spektral_merkez': np.mean(spectral_centroids)
        }
    
    def visualize_waveform(self, output_file='waveform.png'):
        """Ses dalgasını görselleştir"""
        plt.figure(figsize=(12, 4))
        plt.plot(np.linspace(0, len(self.audio)/self.sr, len(self.audio)), self.audio)
        plt.title('Ses Dalgası')
        plt.xlabel('Zaman (saniye)')
        plt.ylabel('Genlik')
        plt.savefig(output_file)
        plt.close()
        
    def visualize_spectrogram(self, output_file='spectrogram.png'):
        """Spektrogramı görselleştir"""
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio)), ref=np.max)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spektrogram')
        plt.savefig(output_file)
        plt.close()
    
    def save_audio(self):
        """Düzeltilmiş sesi kaydet"""
        if self.corrected_audio is not None:
            audio_to_save = self.corrected_audio
        else:
            audio_to_save = self.audio
        sf.write(self.output_file, audio_to_save, self.sr)
        
    def process(self, add_effects=False, speed_factor=1.0, analyze=True, visualize=True):
        """Tüm işlem adımlarını çalıştır"""
        print("Ses dosyası yükleniyor...")
        self.load_audio()
        
        if add_effects:
            print("Efektler ekleniyor...")
            self.add_echo()
            self.add_reverb()
            self.change_speed(speed_factor)
        
        print("Pitch düzeltme işlemi yapılıyor...")
        self.pitch_correction()
        
        if analyze:
            print("\nSes Analizi Yapılıyor...")
            analiz = self.analyze_audio()
            print(f"Ortalama Frekans: {analiz['ortalama_frekans']:.2f} Hz")
            print(f"Maksimum Ses Seviyesi: {analiz['max_ses_seviyesi']:.2f}")
            print(f"Ortalama Ses Seviyesi: {analiz['ortalama_ses_seviyesi']:.2f}")
            print(f"Spektral Merkez: {analiz['spektral_merkez']:.2f} Hz")
        
        if visualize:
            print("\nGörselleştirmeler oluşturuluyor...")
            self.visualize_waveform()
            self.visualize_spectrogram()
        
        print("\nDüzeltilmiş ses kaydediliyor...")
        self.save_audio()
        print(f"İşlem tamamlandı! Sonuç: {self.output_file}")

def main():
    print("""
    ╔═══════════════════════════════════════╗
    ║             AutotuneAI                ║
    ║   Yapay Zeka Destekli Ses İşleme     ║
    ╚═══════════════════════════════════════╝
    """)
    
    input_file = "input.wav"
    output_file = "output_processed.wav"
    
    try:
        processor = AutotuneAI(input_file, output_file)
        processor.process(add_effects=True, speed_factor=1.0, analyze=True, visualize=True)
        
    except FileNotFoundError:
        print("Lütfen 'input.wav' adında bir ses dosyası ekleyin.")
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main() 