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
import torch
import torchcrepe
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter # To load audio if needed by spleeter directly

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
        # Ensure audio is float32 for torchcrepe and pyworld
        audio_temp, self.sr = librosa.load(self.input_file, sr=None, dtype=np.float32)
        # Ensure mono for CREPE
        if audio_temp.ndim > 1:
            audio_temp = librosa.to_mono(audio_temp)
        self.audio = np.ascontiguousarray(audio_temp) # pyworld needs contiguous array

    def apply_autotune(self, strength=1.0, model_capacity='tiny', fmin=50, fmax=550, confidence_threshold=0.4, custom_scale=None):
        """Yapay zeka destekli autotune uygula"""
        if self.audio is None:
            print("Autotune için önce ses dosyası yüklenmeli.")
            return

        print("Autotune işlemi başlatılıyor...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"TorchCREPE {model_capacity} modeli {device} üzerinde çalışacak.")

        # Hop length for CREPE analysis (e.g., 10ms)
        hop_length = int(self.sr * 0.01) # 10ms hop size

        # Ensure audio is a PyTorch tensor
        audio_tensor = torch.tensor(self.audio[np.newaxis, :], dtype=torch.float32, device=device)

        # Get pitch contour (F0) and periodicity using torchcrepe
        # Note: torchcrepe.predict expects audio as a 1D numpy array or path, not a tensor for the predict function.
        # We'll use the lower-level functions torchcrepe.preprocess, torchcrepe.infer, torchcrepe.postprocess

        # Preprocess audio for CREPE
        # Pad audio to facilitate windowing
        padded_audio = torchcrepe.filter.pad(self.audio, hop_length)
        frames = torchcrepe.preprocess.frames(padded_audio, hop_length)
        frames = frames.to(device)

        # Infer pitch probabilities
        # Assuming a batch size, but here we process one file
        probabilities = torchcrepe.infer(frames, model=model_capacity)

        # Postprocess to get pitch and periodicity
        # Using viterbi decoding by default in postprocess
        pitch, periodicity = torchcrepe.postprocess.decode(probabilities, fmin, fmax, model=model_capacity, return_periodicity=True)

        # Squeeze batch dimension if present (predict might add it)
        pitch = pitch.squeeze(0).cpu().numpy()
        periodicity = periodicity.squeeze(0).cpu().numpy()

        # Filter pitch based on periodicity
        periodicity = torchcrepe.filter.median(periodicity, 3) # Smooth periodicity
        pitch = torchcrepe.threshold.At(confidence_threshold)(pitch, periodicity) # Apply confidence threshold
        pitch = torchcrepe.filter.mean(pitch, 3) # Smooth pitch

        # Convert F0 to MIDI notes
        midi_notes = librosa.hz_to_midi(pitch)

        # For notes where pitch is defined (not NaN from thresholding)
        voiced_indices = ~np.isnan(midi_notes)

        # Target MIDI notes (chromatic correction for now)
        # Apply strength: 0 = no change, 1 = full correction
        target_midi_notes = np.copy(midi_notes)
        if custom_scale:
            # Advanced: Snap to a custom scale (list of MIDI note numbers mod 12)
            # For each voiced note, find the closest note in the custom scale
            for i in np.where(voiced_indices)[0]:
                current_note_mod = midi_notes[i] % 12
                scale_note_distances = [min(abs(current_note_mod - s_note), 12 - abs(current_note_mod - s_note)) for s_note in custom_scale]
                closest_scale_note_idx = np.argmin(scale_note_distances)
                target_chroma = custom_scale[closest_scale_note_idx]

                original_octave_midi = midi_notes[i]
                original_chroma = original_octave_midi % 12

                # Adjust target_chroma to be the one closest to original_chroma
                # e.g. if original is 11.8 (B) and scale has 0 (C) and 10 (Bb)
                # if original_chroma is 11.8, target_chroma could be 0 or 10.
                # We want to choose the one that makes original_octave_midi closer to target_octave_midi

                # Form two candidates for target MIDI: one in the same octave, one in adjacent if necessary
                target_midi_candidate1 = np.floor(original_octave_midi / 12) * 12 + target_chroma
                target_midi_candidate2 = np.ceil(original_octave_midi / 12) * 12 + target_chroma

                # Choose the candidate closer to the original MIDI note
                if abs(original_octave_midi - target_midi_candidate1) <= abs(original_octave_midi - target_midi_candidate2):
                    rounded_midi = target_midi_candidate1
                else:
                    rounded_midi = target_midi_candidate2

                # If the closest scale note is further than a semitone away in the other direction, adjust octave
                if abs(original_chroma - target_chroma) > 6: # target is across the octave boundary
                    if original_chroma > target_chroma: # e.g. original C, target B (prev octave)
                        rounded_midi +=12
                    else: # e.g. original B, target C (next octave)
                        rounded_midi -=12

                target_midi_notes[i] = original_octave_midi * (1.0 - strength) + rounded_midi * strength

        else: # Chromatic correction
            rounded_midi_notes = np.round(midi_notes[voiced_indices])
            target_midi_notes[voiced_indices] = midi_notes[voiced_indices] * (1.0 - strength) + rounded_midi_notes * strength

        # Convert target MIDI notes back to F0
        corrected_f0 = librosa.midi_to_hz(target_midi_notes)
        # Fill NaN values (unvoiced / low confidence) with 0 Hz for PyWorld, or original pitch if preferred
        corrected_f0[np.isnan(corrected_f0)] = 0

        # Ensure corrected_f0 is double for PyWorld
        corrected_f0 = corrected_f0.astype(np.double)

        # PyWorld analysis (spectral envelope and aperiodicity from original audio)
        # Need to use the original audio for analysis to keep its timbre
        # Ensure self.audio is double for pyworld
        audio_double = self.audio.astype(np.double)
        _f0, _time = pw.dio(audio_double, self.sr, frame_period=hop_length/self.sr*1000) # frame_period in ms
        _sp = pw.cheaptrick(audio_double, _f0, _time, self.sr)
        _ap = pw.d4c(audio_double, _f0, _time, self.sr)

        # Align CREPE F0 length with PyWorld F0 length
        # PyWorld's _f0 length is often different from CREPE's due to different framing/padding.
        # We need to resample/interpolate corrected_f0 to match the length of _f0.
        if len(corrected_f0) != len(_f0):
            print(f"Aligning F0 lengths: CREPE F0 len {len(corrected_f0)}, PyWorld F0 len {len(_f0)}")
            # Simple resampling using numpy.interp
            # Create x-axis for corrected_f0 and _f0
            x_corrected_f0 = np.linspace(0, 1, len(corrected_f0))
            x_pyworld_f0 = np.linspace(0, 1, len(_f0))
            corrected_f0_aligned = np.interp(x_pyworld_f0, x_corrected_f0, corrected_f0)
        else:
            corrected_f0_aligned = corrected_f0

        # Synthesize audio with corrected F0 and original SP/AP
        # Ensure f0 is C-contiguous and double
        corrected_f0_aligned = np.ascontiguousarray(corrected_f0_aligned, dtype=np.double)
        
        print("PyWorld syntesizing...")
        self.audio = pw.synthesize(corrected_f0_aligned, _sp, _ap, self.sr, frame_period=hop_length/self.sr*1000)
        self.audio = self.audio.astype(np.float32) # Back to float32
        print("Autotune işlemi tamamlandı.")

    def apply_source_separation(self, model_name='spleeter:2stems', output_stems_path='separated_stems'):
        """Kaynak ayırma (source separation) uygula"""
        if not self.input_file or not os.path.exists(self.input_file):
            print("Kaynak ayırma için geçerli bir giriş dosyası bulunamadı.")
            return None

        print(f"Kaynak ayırma işlemi '{model_name}' modeli ile başlatılıyor...")
        try:
            separator = Separator(model_name)

            # Define where Spleeter will save the stems.
            # It creates a subdirectory named after the input file inside output_stems_path.
            # e.g., output_stems_path/input_filename_without_ext/vocals.wav
            if not os.path.exists(output_stems_path):
                os.makedirs(output_stems_path, exist_ok=True)

            print(f"Stems '{output_stems_path}' dizinine kaydedilecek.")
            separator.separate_to_file(self.input_file, output_stems_path)

            # Determine the names of the output files/stems
            input_filename_without_ext = os.path.splitext(os.path.basename(self.input_file))[0]
            stem_output_folder = os.path.join(output_stems_path, input_filename_without_ext)

            generated_stems = {}
            if model_name == 'spleeter:2stems':
                possible_stems = ['vocals', 'accompaniment']
            elif model_name == 'spleeter:4stems':
                possible_stems = ['vocals', 'drums', 'bass', 'other']
            elif model_name == 'spleeter:5stems':
                possible_stems = ['vocals', 'drums', 'bass', 'piano', 'other']
            else: # Fallback for custom models, though not explicitly supported yet
                possible_stems = ['vocals', 'drums', 'bass', 'piano', 'other', 'accompaniment']

            for stem_name in possible_stems:
                stem_file_path = os.path.join(stem_output_folder, f"{stem_name}.wav")
                if os.path.exists(stem_file_path):
                    generated_stems[stem_name] = stem_file_path

            print(f"Kaynak ayırma tamamlandı. Stems: {generated_stems}")
            return generated_stems

        except Exception as e:
            print(f"Kaynak ayırma sırasında bir hata oluştu: {e}")
            # Potentially log more details or raise specific exceptions
            return None

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
                     beat_slice=False, add_sidechain=False, apply_autotune=False,
                     autotune_strength=0.8, autotune_model='tiny', autotune_confidence=0.4,
                     apply_source_separation=False, spleeter_model='spleeter:2stems',
                     stems_output_dir='separated_stems_output'):
        """Gelişmiş remix işlem pipeline'ı"""

        # Source separation is done first if enabled, on the original input file
        if apply_source_separation:
            print("Kaynak ayırma (Spleeter) işlemi uygulanıyor...")
            # Ensure a valid input file is set for Spleeter, even if self.audio is loaded from it
            if not self.input_file or not os.path.exists(self.input_file):
                 print("Uyarı: Kaynak ayırma için giriş dosyası bulunamadı, bu adım atlanıyor.")
            else:
                self.apply_source_separation(model_name=spleeter_model, output_stems_path=stems_output_dir)
                # Note: This currently only saves stems. It doesn't change self.audio for further processing.
                # The user would need to use the saved stems manually.
                # If the goal was to process a specific stem (e.g. vocals), the workflow would need adjustment:
                # 1. Separate. 2. User selects a stem. 3. Load that stem into self.audio. 4. Process.

        print("Ses dosyası yükleniyor (ana işlem için)...")
        self.load_audio() # This loads self.input_file into self.audio

        if apply_autotune:
            print("Autotune uygulanıyor...")
            # For now, only chromatic autotune. Scale selection can be added later.
            self.apply_autotune(strength=autotune_strength, model_capacity=autotune_model, confidence_threshold=autotune_confidence, custom_scale=None)
        
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
            add_sidechain=True,    # Sidechain kompresyon aktif
            apply_autotune=True,   # Autotune aktif
            autotune_strength=0.7, # Autotune gücü
            autotune_model='tiny', # Autotune CREPE modeli
            apply_source_separation=True, # Kaynak ayırmayı aktif et
            spleeter_model='spleeter:2stems', # Kullanılacak Spleeter modeli
            stems_output_dir='cli_separated_stems' # CLI test için çıktı klasörü
        )
        
    except FileNotFoundError:
        print("Lütfen 'input.wav' adında bir ses dosyası ekleyin.")
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()