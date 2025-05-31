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

    def add_reverb(self, decay_time=0.5, damping=0.5, wet_dry_mix=0.3):
        """Basit algoritmik reverb efekti (Schroeder Reverberator benzeri)"""
        if wet_dry_mix == 0:
            return

        num_samples = len(self.audio)
        dry_signal = self.audio.copy()
        # wet_signal = np.zeros(num_samples) # Kullanılmıyor gibi, comb_outputs'tan başlıyoruz

        # Schroeder Reverb için genel parametreler
        comb_delay_times_sec = [0.0297, 0.0371, 0.0411, 0.0437, 0.022, 0.033] # Biraz daha çeşitlilik
        allpass_delay_times_sec = [0.005, 0.0017, 0.007, 0.0023] # Daha fazla all-pass
        allpass_feedback = 0.5

        from scipy.signal import lfilter # Başa alalım

        # Paralel Comb Filtreler
        comb_outputs_sum = np.zeros(num_samples)
        for delay_sec in comb_delay_times_sec:
            delay_samples = int(delay_sec * self.sr)
            if delay_samples <= 0 or delay_samples >= num_samples: continue

            feedback_gain = 0.0 # Default
            if decay_time > 0:
                 # RT60 formülü: g = 0.001^(D/RT60)
                 # Veya daha basit bir üssel azalma:
                 feedback_gain = np.exp(-2.2 * delay_sec / decay_time)

            feedback_gain *= (1 - (damping * (delay_sec / 0.05))) # Damping frekansa bağlı olsun (kısa delay'ler daha az sönümlenir)
            feedback_gain = np.clip(feedback_gain, 0, 0.98) # Gain'in çok artmasını engelle

            b_comb = [1.0]
            a_comb = np.zeros(delay_samples + 1)
            a_comb[0] = 1.0
            a_comb[delay_samples] = -feedback_gain

            comb_output_single = lfilter(b_comb, a_comb, self.audio)
            comb_outputs_sum += comb_output_single
        
        # Comb filtre çıkışlarını normalize etmiyoruz, ortalamasını alabiliriz veya doğrudan toplarız.
        # Genellikle doğrudan toplanır ve sonra genel seviye ayarlanır.
        # wet_signal = comb_outputs_sum / len(comb_delay_times_sec) # Ortalama almak yerine doğrudan toplama daha yaygın
        wet_signal = comb_outputs_sum / np.sqrt(len(comb_delay_times_sec)) # Enerjiyi korumak için RMS benzeri normalizasyon


        # Seri All-Pass Filtreler
        allpass_processed_signal = wet_signal # Comb çıkışını al
        for delay_sec in allpass_delay_times_sec:
            delay_samples = int(delay_sec * self.sr)
            if delay_samples <= 0 or delay_samples >= num_samples: continue

            g_ap = allpass_feedback # All-pass gain'i genellikle sabit tutulur
                                    # damping burada da etkili olabilir ama şimdilik sabit.

            b_ap_schroeder = np.zeros(delay_samples + 1)
            b_ap_schroeder[0] = -g_ap
            b_ap_schroeder[delay_samples] = 1.0

            a_ap_schroeder = np.zeros(delay_samples + 1)
            a_ap_schroeder[0] = 1.0
            a_ap_schroeder[delay_samples] = -g_ap # lfilter'da feedback terimi negatif

            allpass_processed_signal = lfilter(b_ap_schroeder, a_ap_schroeder, allpass_processed_signal)

        wet_signal = allpass_processed_signal

        reverbed_audio = (1 - wet_dry_mix) * dry_signal + wet_dry_mix * wet_signal

        # Clipping'i önlemek için normalizasyon
        max_abs_val = np.max(np.abs(reverbed_audio))
        if max_abs_val > 0.99: # 0.99 sınırı, biraz headroom bırakmak için
            reverbed_audio = reverbed_audio / max_abs_val * 0.99

        self.audio = reverbed_audio

    def add_filter(self, filter_type='lowpass', cutoff_freq=1000, order=4, q_factor=1.0):
        """Gelişmiş filtre sistemi (lowpass, highpass, bandpass, bandstop destekler)"""
        nyquist = self.sr * 0.5

        if filter_type in ['lowpass', 'highpass']:
            if not isinstance(cutoff_freq, (int, float)) or cutoff_freq <= 0 or cutoff_freq >= nyquist:
                print(f"Uyarı: {filter_type} için cutoff_freq ({cutoff_freq} Hz) (0, {nyquist} Hz) aralığında geçerli bir sayı olmalı. Filtre uygulanmayacak.")
                return
            normalized_cutoff = cutoff_freq / nyquist
            # Kesim frekansının çok küçük veya 1'e çok yakın olmamasını sağla (nümerik stabilite için)
            normalized_cutoff = np.clip(normalized_cutoff, 1e-6, 1.0 - 1e-6)
            b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
        elif filter_type in ['bandpass', 'bandstop']:
            if not isinstance(cutoff_freq, (int, float)) or cutoff_freq <= 0 or cutoff_freq >= nyquist:
                 print(f"Uyarı: {filter_type} için merkez frekans ({cutoff_freq} Hz) (0, {nyquist} Hz) aralığında geçerli bir sayı olmalı. Filtre uygulanmayacak.")
                 return
            if not isinstance(q_factor, (int,float)) or q_factor <= 0:
                print(f"Uyarı: {filter_type} için Q faktörü ({q_factor}) pozitif bir sayı olmalı. Filtre uygulanmayacak.")
                return

            center_freq_norm = cutoff_freq / nyquist
            bandwidth_norm = center_freq_norm / q_factor

            low_cutoff_norm = center_freq_norm - (bandwidth_norm / 2)
            high_cutoff_norm = center_freq_norm + (bandwidth_norm / 2)

            if low_cutoff_norm >= high_cutoff_norm: # Bant genişliği çok büyükse veya merkez frekans kenara çok yakınsa olabilir
                 print(f"Uyarı: {filter_type} için hesaplanan kesim frekansları geçersiz (low_norm: {low_cutoff_norm:.3f}, high_norm: {high_cutoff_norm:.3f}). Q veya merkez frekansını ayarlayın. Filtre uygulanmayacak.")
                 return

            # Normalize edilmiş frekansları (0, 1) aralığına klipleyelim (açık aralık gibi düşünülmeli)
            low_cutoff_norm = np.clip(low_cutoff_norm, 1e-6, 1.0 - 1e-6)
            high_cutoff_norm = np.clip(high_cutoff_norm, 1e-6, 1.0 - 1e-6)

            if low_cutoff_norm >= high_cutoff_norm: # Kliplemeden sonra tekrar kontrol
                 print(f"Uyarı: {filter_type} için Kliplenmiş kesim frekansları geçersiz (low_norm: {low_cutoff_norm:.3f}, high_norm: {high_cutoff_norm:.3f}). Filtre uygulanmayacak.")
                 return

            b, a = butter(order, [low_cutoff_norm, high_cutoff_norm], btype=filter_type, analog=False)
        else:
            print(f"Geçersiz filtre tipi: {filter_type}. Desteklenen tipler: 'lowpass', 'highpass', 'bandpass', 'bandstop'. Filtre uygulanmayacak.")
            return

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
                     beat_slice=False, add_sidechain=False,
                     reverb_decay_time=0.0, reverb_damping=0.5, reverb_mix=0.0, # Reverb parametreleri
                     eq_filter_type=None, eq_center_freq=1000, eq_q_factor=1.0, eq_order=4 # EQ Filtre parametreleri
                     ):
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
            self.add_delay(0.3, 0.4, 0.3) # Bu efektler `add_effects` bayrağına bağlı kalabilir
            self.add_flanger(0.7, 0.003, 0.4)
            # self.add_filter('lowpass', 2000, order=4) # Eski genel filtreyi kaldırıyoruz, EQ ile yönetilecek

        # Yeni EQ Filtre Uygulaması (add_effects'ten bağımsız olabilir veya ona bağlanabilir)
        # Şimdilik add_effects'ten bağımsız olarak çalışsın.
        if eq_filter_type:
            print(f"{eq_filter_type} EQ filtresi uygulanıyor (Freq: {eq_center_freq} Hz, Q: {eq_q_factor}, Order: {eq_order})...")
            self.add_filter(filter_type=eq_filter_type,
                            cutoff_freq=eq_center_freq,
                            order=eq_order,
                            q_factor=eq_q_factor)

        # Reverb Uygulaması (add_effects'ten bağımsız olabilir veya ona bağlanabilir)
        # Şimdilik add_effects'ten bağımsız olarak çalışsın.
        if reverb_mix > 0: # reverb_mix > 0 ise reverb uygula
            print(f"Reverb ekleniyor (Decay: {reverb_decay_time}s, Damping: {reverb_damping}, Mix: {reverb_mix})...")
            self.add_reverb(decay_time=reverb_decay_time,
                            damping=reverb_damping,
                            wet_dry_mix=reverb_mix)
            
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
            tempo_change=1.0,      # Tempo değişikliği yok
            pitch_steps=0,         # Perde değişikliği yok
            add_effects=True,      # Temel efektler (delay, flanger) aktif kalsın mı? Evet.
            beat_slice=False,      # Beat slicing kapalı
            add_sidechain=False,   # Sidechain kapalı

            # Yeni Reverb Ayarları
            reverb_decay_time=0.7, # Saniye cinsinden reverb süresi (0 ise reverb yok gibi)
            reverb_damping=0.4,    # Yüksek frekans sönümlemesi (0-1 aralığı, 0: sönümleme yok)
            reverb_mix=0.2,        # Wet/Dry karışımı (0: sadece dry, 1: sadece wet)

            # Yeni EQ Filtre Ayarları
            eq_filter_type='bandpass', # Seçenekler: 'lowpass', 'highpass', 'bandpass', 'bandstop', None
            eq_center_freq=1200,   # Hz cinsinden merkez/kesim frekansı
            eq_q_factor=1.0,       # Q faktörü (bandpass/bandstop için anlamlı, >0 olmalı)
            eq_order=3             # Filtre derecesi (örn: 2, 3, 4...)
        )
        
        # Örnek 2: Sadece highpass filtre ve biraz daha uzun reverb
        # output_file_hp_reverb = "output_remix_hp_reverb.wav"
        # remixer_hp_reverb = Premiermixx(input_file, output_file_hp_reverb)
        # print(f"\nİkinci remix oluşturuluyor: {output_file_hp_reverb}")
        # remixer_hp_reverb.process_remix(
        #     add_effects=False, # Temel delay/flanger kapalı
        #     reverb_decay_time=1.5,
        #     reverb_damping=0.6,
        #     reverb_mix=0.3,
        #     eq_filter_type='highpass',
        #     eq_center_freq=300, # Düşük frekansları kes
        #     eq_order=4
        # )

    except FileNotFoundError:
        print("Lütfen 'input.wav' adında bir ses dosyası ekleyin.")
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()