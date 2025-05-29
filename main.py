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
from scipy.signal import butter, filtfilt

class Premiermixx:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.audio = None
        self.sr = None
        self.processed_audio = None
        self.beat_frames = None
        self.tempo = None
        
    def load_audio(self):
        """Ses dosyasını yükle"""
        self.audio, self.sr = librosa.load(self.input_file, sr=None)
        
    def apply_beat_detection(self):
        """Ritim tespiti ve senkronizasyonu"""
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.audio, sr=self.sr)
        beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
        return self.tempo, beat_times
    
    def time_stretch(self, rate=1.0):
        """Zaman uzatma/kısaltma efekti"""
        self.audio = librosa.effects.time_stretch(self.audio, rate=rate)
        
    def pitch_shift(self, steps):
        """Perde kaydırma"""
        self.audio = librosa.effects.pitch_shift(self.audio, sr=self.sr, n_steps=steps)
        
    def add_delay(self, delay_time=0.3, decay=0.5, feedback=0.3):
        """Gelişmiş Delay efekti"""
        delay_samples = int(self.sr * delay_time)
        num_echoes = int(1 / decay)
        delay = np.zeros_like(self.audio)
        
        for i in range(num_echoes):
            echo_delay = delay_samples * (i + 1)
            if echo_delay < len(self.audio):
                echo = np.zeros_like(self.audio)
                echo[echo_delay:] = self.audio[:-echo_delay] * (decay ** i) * feedback
                delay += echo
                
        self.audio = self.audio + delay
        
    def add_filter(self, filter_type='lowpass', cutoff_freq=1000, order=4):
        """Gelişmiş filtre sistemi"""
        nyquist = self.sr * 0.5
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normalized_cutoff, btype=filter_type)
        self.audio = filtfilt(b, a, self.audio)
            
    def add_flanger(self, rate=0.5, depth=0.002, feedback=0.5):
        """Gelişmiş Flanger efekti"""
        t = np.arange(len(self.audio)) / self.sr
        mod = np.sin(2 * np.pi * rate * t)
        delay_samples = int(depth * self.sr)
        flanger = np.zeros_like(self.audio)
        
        for i in range(len(self.audio)):
            delay = int(delay_samples * (1 + mod[i]))
            if i >= delay:
                flanger[i] = self.audio[i - delay]
        
        self.audio = (1 - feedback) * self.audio + feedback * flanger
        
    def create_loop(self, start_time, end_time, repeats=4, crossfade=0.1):
        """Gelişmiş loop oluşturma"""
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        crossfade_samples = int(crossfade * self.sr)
        
        loop_section = self.audio[start_sample:end_sample]
        loop_length = len(loop_section)
        
        # Crossfade için fade in/out eğrileri
        fade_in = np.linspace(0, 1, crossfade_samples)
        fade_out = np.linspace(1, 0, crossfade_samples)
        
        # Loop oluşturma
        final_loop = np.zeros(loop_length * repeats)
        for i in range(repeats):
            start_idx = i * loop_length
            end_idx = start_idx + loop_length
            final_loop[start_idx:end_idx] = loop_section
            
            # Crossfade uygulama
            if i > 0:
                final_loop[start_idx:start_idx+crossfade_samples] *= fade_in
                final_loop[start_idx-crossfade_samples:start_idx] *= fade_out
                
        return final_loop
    
    def beat_slice(self, slice_length=4):
        """Beat bazlı kesme ve yeniden düzenleme"""
        if self.beat_frames is None:
            self.apply_beat_detection()
            
        # Beat'leri slice_length kadar gruplara ayır
        beat_samples = librosa.frames_to_samples(self.beat_frames)
        num_slices = len(beat_samples) // slice_length
        slices = []
        
        for i in range(num_slices):
            start = beat_samples[i * slice_length]
            end = beat_samples[min((i + 1) * slice_length, len(beat_samples) - 1)]
            slice_audio = self.audio[start:end]
            slices.append(slice_audio)
            
        # Sliceları karıştır
        np.random.shuffle(slices)
        
        # Sliceları birleştir
        self.audio = np.concatenate(slices)
    
    def add_sidechain(self, threshold=-20, ratio=4, attack=0.01, release=0.1):
        """Sidechain kompresyon efekti"""
        if self.beat_frames is None:
            self.apply_beat_detection()
            
        beat_samples = librosa.frames_to_samples(self.beat_frames)
        envelope = np.ones_like(self.audio)
        
        attack_samples = int(attack * self.sr)
        release_samples = int(release * self.sr)
        
        for beat in beat_samples:
            if beat + release_samples < len(envelope):
                envelope[beat:beat+attack_samples] = np.linspace(1, 1/ratio, attack_samples)
                envelope[beat+attack_samples:beat+release_samples] = np.linspace(1/ratio, 1, release_samples-attack_samples)
                
        self.audio *= envelope
    
    def visualize_remix(self, output_file='remix_analysis.png'):
        """Gelişmiş remix görselleştirmesi"""
        plt.figure(figsize=(15, 12))
        
        # Dalga formu
        plt.subplot(3, 1, 1)
        plt.plot(np.linspace(0, len(self.audio)/self.sr, len(self.audio)), self.audio)
        plt.title('Remix Dalga Formu')
        plt.xlabel('Zaman (saniye)')
        plt.ylabel('Genlik')
        
        # Spektrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio)), ref=np.max)
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Remix Spektrogramı')
        
        # Beat tracking görselleştirmesi
        if self.beat_frames is not None:
            plt.subplot(3, 1, 3)
            onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
            times = librosa.times_like(onset_env, sr=self.sr)
            plt.plot(times, onset_env, label='Onset strength')
            plt.vlines(librosa.frames_to_time(self.beat_frames), 0, onset_env.max(), 
                      color='r', alpha=0.5, label='Beats')
            plt.title('Beat Tracking')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    def save_remix(self):
        """Remixlenmiş sesi kaydet"""
        sf.write(self.output_file, self.audio, self.sr)
        
    def process_remix(self, tempo_change=1.0, pitch_steps=0, add_effects=True, 
                     beat_slice=False, add_sidechain=False):
        """Gelişmiş remix işlem pipeline'ı"""
        print("Ses dosyası yükleniyor...")
        self.load_audio()
        
        print("Tempo ve ritim analizi yapılıyor...")
        self.apply_beat_detection()
        
        if beat_slice:
            print("Beat slicing uygulanıyor...")
            self.beat_slice(slice_length=4)
        
        if tempo_change != 1.0:
            print(f"Tempo değiştiriliyor ({tempo_change}x)...")
            self.time_stretch(tempo_change)
            
        if pitch_steps != 0:
            print(f"Perde kaydırma uygulanıyor ({pitch_steps} adım)...")
            self.pitch_shift(pitch_steps)
            
        if add_effects:
            print("Efektler ekleniyor...")
            self.add_delay(0.3, 0.4, 0.3)
            self.add_flanger(0.7, 0.003, 0.4)
            self.add_filter('lowpass', 2000, order=4)
            
        if add_sidechain:
            print("Sidechain kompresyon uygulanıyor...")
            self.add_sidechain()
        
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
            add_effects=True,      # Efektleri ekle
            beat_slice=True,       # Beat slicing aktif
            add_sidechain=True     # Sidechain kompresyon aktif
        )
        
    except FileNotFoundError:
        print("Lütfen 'input.wav' adında bir ses dosyası ekleyin.")
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()