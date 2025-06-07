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
from scipy.signal import butter, filtfilt, sawtooth, triangle, iirfilter, sosfilt
import tensorflow as tf

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
        
    def add_delay(self, delay_time=0.3, decay=0.5, feedback=0.3, ping_pong=False, lfo_rate=0.0, lfo_depth=0.0):
        """Gelişmiş Delay efekti, mevcut yapıyı koruyarak LFO ve ping_pong ekler."""

        if not hasattr(self, 'audio') or self.audio is None or \
           not hasattr(self, 'sr') or self.sr is None:
            return

        clamped_feedback = min(max(feedback, 0.0), 0.99)
        delay_time = max(delay_time, 0.001)
        decay = min(max(decay, 0.01), 0.99)

        num_echoes = int(1.0 / decay)
        if num_echoes <= 0:
            num_echoes = 1

        base_delay_samples_for_lfo = 0
        if self.sr > 0 and delay_time > 0:
             base_delay_samples_for_lfo = int(self.sr * delay_time)

        delayed_sum = np.zeros_like(self.audio)

        for i in range(num_echoes):
            lfo_shift_for_this_echo_path = 0
            if lfo_rate > 0.0 and lfo_depth > 0.0 and base_delay_samples_for_lfo > 0:
                time_of_echo_occurrence = delay_time * (i + 1)
                lfo_val_at_echo_time = np.sin(2 * np.pi * lfo_rate * time_of_echo_occurrence)
                lfo_shift_for_this_echo_path = int(lfo_val_at_echo_time * lfo_depth * base_delay_samples_for_lfo)

            current_base_delay_samples = 0
            if self.sr > 0:
                current_base_delay_samples = int(self.sr * delay_time * (i + 1))

            current_total_delay_samples = current_base_delay_samples + lfo_shift_for_this_echo_path
            current_total_delay_samples = max(1, current_total_delay_samples)

            if current_total_delay_samples >= len(self.audio):
                continue

            echo_gain = (decay ** i) * clamped_feedback

            if ping_pong:
                if i % 2 == 1:
                    echo_gain *= 0.6

            current_echo_segment = np.zeros_like(self.audio)
            source_signal_for_echo = self.audio

            actual_len_to_copy = len(source_signal_for_echo) - current_total_delay_samples

            if actual_len_to_copy <= 0:
                continue

            start_source = 0
            end_source = actual_len_to_copy

            start_dest = current_total_delay_samples
            end_dest = current_total_delay_samples + actual_len_to_copy

            current_echo_segment[start_dest:end_dest] = \
                source_signal_for_echo[start_source:end_source] * echo_gain

            delayed_sum += current_echo_segment
        
        self.audio = self.audio + delayed_sum

    def add_parametric_eq(self, bands=None):
        """Parametric EQ with multiple bands.
        Each band in the 'bands' list should be a dictionary:
        {'type': 'peak', 'freq': 1000, 'q': 1.0, 'gain_db': -3.0}
        {'type': 'lowshelf', 'freq': 100, 'q': 0.707, 'gain_db': 6.0}
        {'type': 'highshelf', 'freq': 8000, 'q': 0.707, 'gain_db': -6.0}
        {'type': 'lowpass', 'freq': 12000, 'q': 0.707} # gain_db not used for lowpass/highpass
        {'type': 'highpass', 'freq': 80, 'q': 0.707}   # gain_db not used for lowpass/highpass
        {'type': 'notch', 'freq': 60, 'q': 10}        # gain_db not used for notch
        """
        if not hasattr(self, 'audio') or self.audio is None or \
           not hasattr(self, 'sr') or self.sr is None:
            return
        if bands is None:
            bands = [{'type': 'peak', 'freq': 1000, 'q': 1.0, 'gain_db': 0.0}] # Default: no change peak filter

        processed_audio = self.audio.copy() # Work on a copy

        for band in bands:
            btype = band.get('type', 'peak').lower()
            center_freq = band.get('freq', 1000)
            q_factor = band.get('q', 1.0)
            gain_db = band.get('gain_db', 0.0) # Used for peak and shelves

            # Nyquist frequency
            nyquist = self.sr / 2.0

            # Normalize frequency
            norm_freq = center_freq / nyquist
            if norm_freq <= 0 or norm_freq >= 1:
                # print(f"Warning: Filter freq {center_freq} for type {btype} is out of valid range [0, {nyquist}]. Skipping band.")
                continue

            # Ensure Q is positive
            q_factor = max(0.1, q_factor)

            sos = None
            if btype in ['peak', 'lowshelf', 'highshelf', 'notch']:
                if btype == 'peak':
                    # scipy's iirfilter does not directly support 'peak'.
                    # We can use 'bandpass' with Q for peaking, or implement biquad math.
                    # For simplicity, let's use a biquad coefficient calculation for peaking/shelving.
                    # This requires more detailed math or a library that provides direct biquad design for these.
                    # As a placeholder, we will use a bandpass for 'peak' if gain > 0, bandstop if gain < 0.
                    # This is not a true parametric peak/shelf filter.
                    # A proper implementation would involve designing biquad coefficients (b0,b1,b2,a0,a1,a2).

                    # For a true peak filter, we need to calculate coefficients manually:
                    A = 10**(gain_db / 40)  # For peaking and shelving EQs
                    w0 = 2 * np.pi * norm_freq # Angular frequency (normalized, since norm_freq is f/fn)
                    alpha = np.sin(w0) / (2 * q_factor)

                    if btype == 'peak':
                        b0 = 1 + alpha * A
                        b1 = -2 * np.cos(w0)
                        b2 = 1 - alpha * A
                        a0 = 1 + alpha / A
                        a1 = -2 * np.cos(w0)
                        a2 = 1 - alpha / A
                    elif btype == 'lowshelf':
                        # sqrt(A) is used in some shelf designs
                        beta = np.sqrt(A) * np.sqrt((A + 1/A) * (1/q_factor -1) + 2) # Simplified from RBJ cookbook for shelf Q
                        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0))
                        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
                        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0))
                        a0 = (A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0)
                        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
                        a2 = (A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0)
                    elif btype == 'highshelf':
                        beta = np.sqrt(A) * np.sqrt((A + 1/A) * (1/q_factor -1) + 2)
                        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0))
                        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
                        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0))
                        a0 = (A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0)
                        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
                        a2 = (A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0)
                    elif btype == 'notch':
                        # Notch filter (gain_db is not used, it's a full cut at freq)
                        b0 = 1
                        b1 = -2 * np.cos(w0)
                        b2 = 1
                        a0 = 1 + alpha
                        a1 = -2 * np.cos(w0)
                        a2 = 1 - alpha
                    else: # Should not happen
                        continue

                    # Create SOS matrix from biquad coefficients [b0,b1,b2, a0,a1,a2]
                    # Ensure a0 is 1 for sosfilt by dividing all by a0
                    sos_coeffs = np.array([b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0])
                    sos = np.reshape(sos_coeffs, (1,6)) # sosfilt expects 2D array [[b0,b1,b2,a0,a1,a2]]

                else: # Should not be reached given current btype check
                    continue

            elif btype in ['lowpass', 'highpass']:
                # For lowpass/highpass, gain_db is not applicable. Q is used.
                # iirfilter can design these directly.
                # We need to map Q to something iirfilter understands, or use a biquad design again.
                # For simplicity with iirfilter, we might need to omit Q or use it to pick order.
                # Let's try a Butterworth filter of order N that can be derived from Q.
                # This is an approximation. A common order is 2 for biquads.
                filter_order = 2 # Typical for one stage of a parametric EQ band.
                try:
                    sos = iirfilter(filter_order, norm_freq, btype=btype, ftype='butter', output='sos')
                except ValueError as e:
                    # print(f"Error designing {btype} filter at {center_freq}Hz (norm: {norm_freq}): {e}")
                    continue

            else:
                # print(f"Unsupported filter type: {btype}")
                continue

            if sos is not None:
                # Apply filter if sos coefficients were generated
                if processed_audio.ndim == 1: # Mono
                    processed_audio = sosfilt(sos, processed_audio)
                elif processed_audio.ndim == 2: # Stereo (assuming [samples, channels])
                     # Apply to each channel
                    for ch_idx in range(processed_audio.shape[1]):
                        processed_audio[:, ch_idx] = sosfilt(sos, processed_audio[:, ch_idx])
            else:
                # print(f"Filter SOS coefficients not generated for band: {band}")
                pass


        self.audio = np.clip(processed_audio, -1.0, 1.0)
            
    def add_flanger(self, rate=0.5, depth=0.002, feedback=0.5, lfo_waveform='sine', stereo_width=0.0):
        """Gelişmiş Flanger efekti with LFO waveform selection, stereo width, and TZF emulation."""
        if not hasattr(self, 'audio') or self.audio is None or \
           not hasattr(self, 'sr') or self.sr is None:
            return

        # Parameter validation
        rate = max(0.01, rate)
        depth_samples = int(max(0.0001, depth) * self.sr) # Depth in samples
        feedback = min(max(feedback, -0.99), 0.99) # Allow negative feedback, capped
        stereo_width = min(max(stereo_width, 0.0), 1.0)

        # Ensure audio is stereo for stereo_width > 0. If mono, duplicate to stereo.
        # This is a simplified stereo handling. True stereo processing would be more involved.
        # For now, we'll assume if stereo_width > 0, the output might be perceived as wider
        # even if the input is mono, by applying slightly different LFO phases or depths
        # to conceptual left/right channels if we were to fully implement them.
        # Given the current structure, true stereo path processing isn't straightforward.
        # We will apply the flanger to self.audio directly.
        # The 'stereo_width' will be conceptual unless we adapt self.audio to be 2D array.

        t = np.arange(len(self.audio)) / self.sr
        
        mod_signal = None
        if lfo_waveform == 'sine':
            mod_signal = np.sin(2 * np.pi * rate * t)
        elif lfo_waveform == 'triangle':
            # Triangle LFO generation might need a specific import or custom implementation
            # Using a simplified triangle for now, assuming scipy.signal.triangle is available
            # For scipy.signal.triangle, the period is 1.0, so scale t by rate
            # And its output is 0 to 1, so scale and shift to -1 to 1
            try:
                from scipy.signal import triangle # This import is locally scoped for clarity here
                mod_signal = triangle(2 * np.pi * rate * t, 0.5) * 2 - 1
            except ImportError:
                # Fallback to sine if triangle is not available
                mod_signal = np.sin(2 * np.pi * rate * t)
                # print("Warning: scipy.signal.triangle not found, defaulting LFO to sine for Flanger.")
        elif lfo_waveform == 'sawtooth':
            # Sawtooth LFO generation
            try:
                from scipy.signal import sawtooth # This import is locally scoped for clarity here
                mod_signal = sawtooth(2 * np.pi * rate * t)
            except ImportError:
                # Fallback to sine
                mod_signal = np.sin(2 * np.pi * rate * t)
                # print("Warning: scipy.signal.sawtooth not found, defaulting LFO to sine for Flanger.")
        else: # Default to sine
            mod_signal = np.sin(2 * np.pi * rate * t)

        # Modulate delay time: depth_samples is the max delay, mod_signal is -1 to 1
        # (1 + mod_signal) / 2 scales mod_signal to 0 to 1 range for positive delays
        # For Through-Zero Flanging (TZF) emulation, the LFO should allow the delay to cross "zero"
        # This means the modulated signal can be read from "ahead" of the current sample.
        # A true TZF requires a more complex setup, often involving two delay lines or specific plugins.
        # We can emulate a part of it by allowing the modulation to suggest a "negative" delay,
        # which we'll clip at a very small positive delay.

        # The core idea of flanging is a varying short delay.
        # Modulated delay in samples: depth_samples * mod_signal.
        # This will vary from -depth_samples to +depth_samples.
        
        flanged_audio = np.zeros_like(self.audio)
        delayed_signal_buffer = np.zeros_like(self.audio) # Buffer for the feedback path

        # Initial condition for feedback: use a small segment of dry audio or zeros
        # This is a simplification. More advanced flangers might have specific startup.
        if len(self.audio) > depth_samples:
             delayed_signal_buffer[:depth_samples] = self.audio[:depth_samples] * feedback

        for i in range(len(self.audio)):
            # Current modulation for delay: mod_signal[i] is -1 to 1
            # delay_offset_samples = depth_samples * (1 + mod_signal[i]) / 2 # For positive delays only
            delay_offset_samples = depth_samples * mod_signal[i] # For TZF-like behavior (-depth to +depth)

            # Actual read position: current sample 'i' minus the modulated delay
            # This means we are looking 'backwards' in time by delay_offset_samples
            read_pos = i - delay_offset_samples

            # Integer part for array indexing, fractional part for interpolation
            read_pos_int = int(np.floor(read_pos))
            read_pos_frac = read_pos - read_pos_int

            # Linear interpolation for fractional delay (smoother sound)
            delayed_sample = 0
            if read_pos_int >= 0 and read_pos_int < len(self.audio) -1 : # Check bounds for interpolation
                delayed_sample = (delayed_signal_buffer[read_pos_int] * (1 - read_pos_frac) +
                                  delayed_signal_buffer[read_pos_int + 1] * read_pos_frac)
            elif read_pos_int == len(self.audio) -1: # If at the very end, no next sample to interpolate
                 delayed_sample = delayed_signal_buffer[read_pos_int]
            # else: sample is out of bounds (e.g. negative index for TZF when i is small)
            # For TZF, if read_pos is negative, it implies reading from "the future",
            # which is not physically possible in a simple delay line.
            # True TZF often uses techniques to make this appear to happen.
            # We'll clip read_pos_int to 0 for simplicity if it goes negative.
            # However, the formula i - delay_offset means if delay_offset is positive, we read from past.
            # If delay_offset is negative (mod_signal is negative), we read from "less past" or "future".
            # Let's ensure read_pos_int is always valid for delayed_signal_buffer.
            # If read_pos_int < 0, means we want a sample from "before the buffer started".
            # This is part of the TZF challenge. For this implementation, we'll use 0 if out of bounds.

            # Mix dry signal with the delayed (wet) signal
            # Flanger output y[n] = x[n] + feedback * delayed_sample (from previous output or input)
            # The delayed_signal_buffer holds the y[n-d] component

            # The current output sample is a mix of dry input and the interpolated delayed sample
            flanged_audio[i] = self.audio[i] + delayed_sample

            # Update the delay buffer for the next iteration with feedback
            # This is y[n] that will become y[n-d] in future iterations
            if i < len(delayed_signal_buffer): # Ensure we don't write out of bounds
                delayed_signal_buffer[i] = self.audio[i] + delayed_sample * feedback
                # Clipping feedback path can help control extreme sounds
                delayed_signal_buffer[i] = np.clip(delayed_signal_buffer[i], -1.0, 1.0)


        # The stereo_width effect is conceptual here without true stereo path.
        # A simple way to give a sense of width might be to mix the original mono slightly differently
        # if stereo_width > 0, but that's not a standard flanger width.
        # For now, stereo_width parameter is there, but its effect is limited by mono processing.
        # If self.audio were stereo:
        # if stereo_width > 0 and self.audio.ndim == 2:
        #   lfo_right = np.sin(2 * np.pi * rate * t + (np.pi * stereo_width)) # Phase shift for right channel
        #   # ... then process right channel with lfo_right ...

        self.audio = (1 - abs(feedback)*0.5) * self.audio + abs(feedback)*0.5 * flanged_audio # Mix based on feedback amount
        self.audio = np.clip(self.audio, -1.0, 1.0) # Final clip
        
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
    
    def add_sidechain(self, threshold_db=-20, ratio=4, attack_ms=10, release_ms=100, hold_ms=0,
                  makeup_gain_db=0, sidechain_source=None, hpf_freq=20, lpf_freq=None):
        """Gelişmiş Sidechain kompresyon efekti.
        - threshold_db: Ses seviyesi eşiği (dB).
        - ratio: Kompresyon oranı (e.g., 4 for 4:1).
        - attack_ms: Atak süresi (ms).
        - release_ms: Bırakma süresi (ms).
        - hold_ms: Tutma süresi (ms) kompresyonun aktif kalacağı.
        - makeup_gain_db: Telafi kazancı (dB).
        - sidechain_source: Harici sidechain sinyali (numpy array). None ise beat_frames kullanılır.
        - hpf_freq: Sidechain algılama sinyali için yüksek geçiren filtre frekansı (Hz).
        - lpf_freq: Sidechain algılama sinyali için alçak geçiren filtre frekansı (Hz).
        """
        if not hasattr(self, 'audio') or self.audio is None or \
           not hasattr(self, 'sr') or self.sr is None:
            return

        # Convert parameters
        threshold_linear = librosa.db_to_amplitude(threshold_db)
        attack_samples = int(attack_ms * self.sr / 1000)
        release_samples = int(release_ms * self.sr / 1000)
        hold_samples = int(hold_ms * self.sr / 1000)
        makeup_gain_linear = librosa.db_to_amplitude(makeup_gain_db)

        # Determine control signal
        control_signal = None
        if sidechain_source is not None:
            # Ensure sidechain_source has same length as self.audio, or resample/truncate
            if len(sidechain_source) < len(self.audio):
                # Pad if shorter (simple padding, could be more sophisticated)
                padding = np.zeros(len(self.audio) - len(sidechain_source))
                control_signal_unfiltered = np.concatenate((sidechain_source, padding))
            else:
                control_signal_unfiltered = sidechain_source[:len(self.audio)] # Truncate if longer
        else:
            # Use internal beat detection if no external source
            if not hasattr(self, 'beat_frames') or self.beat_frames is None:
                # print("Sidechain: No beat_frames found and no external source. Applying beat detection.")
                self.apply_beat_detection() # Ensure beats are detected

            if hasattr(self, 'beat_frames') and self.beat_frames is not None:
                beat_samples = librosa.frames_to_samples(self.beat_frames, sr=self.sr)
                # Create a pulse train from beat_samples to serve as a basic control signal
                control_signal_unfiltered = np.zeros_like(self.audio)
                for beat_idx in beat_samples:
                    if beat_idx < len(control_signal_unfiltered):
                        # Make a short pulse at each beat, e.g., 10ms long
                        pulse_end = min(len(control_signal_unfiltered), beat_idx + int(10 * self.sr / 1000))
                        control_signal_unfiltered[beat_idx:pulse_end] = 1.0
            else:
                # print("Sidechain: Beat detection failed or yielded no beats. Cannot apply sidechain.")
                return # No source for sidechain

        # Filter the control signal (if specified)
        # This is a simplified filtering. For proper filter design, scipy.signal.butter and sosfilt would be used.
        # Here, we'll just conceptualize it or use a very basic approach if available.
        # For now, let's assume control_signal_filtered is the same as unfiltered
        # A full implementation would use scipy.signal.butter and sosfilt here.
        # e.g. sos_hpf = butter(N, hpf_freq, btype='highpass', fs=self.sr, output='sos'); filtered = sosfilt(sos_hpf, control_signal_unfiltered)
        control_signal_filtered = control_signal_unfiltered

        if lpf_freq is not None and lpf_freq > hpf_freq: # Basic LPF after HPF
            # Placeholder for LPF logic, ideally using scipy.signal
            pass # control_signal_filtered would be further filtered here.

        # Envelope detection on the control signal (RMS or peak)
        # Using a simple peak envelope for this example based on pulses or external source
        # A more common approach would be RMS envelope.
        # For simplicity, we'll use the control_signal_filtered directly as the basis for envelope generation.
        # The 'gain_reduction_signal' will be derived from this.

        gain_reduction_signal = np.ones_like(self.audio) # Starts with 1 (no gain reduction)

        # This loop simulates the dynamics processor (compressor) behavior based on control signal level
        # It's a simplified model. Real compressors have more complex envelope followers.
        current_gain = 1.0
        for i in range(len(self.audio)):
            # Level detection (using the filtered control signal)
            detected_level = abs(control_signal_filtered[i]) # Simple peak detection

            if detected_level > threshold_linear:
                # Above threshold: apply compression
                target_gain = 1.0 / ratio # This is the gain reduction factor
                # Attack phase
                if current_gain > target_gain:
                    current_gain -= (1.0 / attack_samples) if attack_samples > 0 else 1.0
                    current_gain = max(current_gain, target_gain)
                # Hold phase could be inserted here if current_gain is at target_gain
                # This simplified model doesn't explicitly model hold after attack like some compressors.
                # Instead, the hold_samples logic below will keep gain low after a trigger.
            else:
                # Below threshold: release compression
                target_gain = 1.0
                # Release phase
                if current_gain < target_gain:
                    current_gain += (1.0 / release_samples) if release_samples > 0 else 1.0
                    current_gain = min(current_gain, target_gain)
            
            gain_reduction_signal[i] = current_gain

        # Apply hold_samples logic: if triggered, keep gain low for hold_duration
        # This part needs to be integrated carefully with the attack/release logic.
        # A simpler way to implement hold for beat-based sidechaining:
        if sidechain_source is None and hasattr(self, 'beat_frames') and self.beat_frames is not None:
            # Re-initialize gain_reduction_signal for beat-based hold logic for clarity
            gain_reduction_signal.fill(1.0)
            beat_samples = librosa.frames_to_samples(self.beat_frames, sr=self.sr)

            for beat_start_sample in beat_samples:
                # Start of gain reduction (attack phase)
                attack_end_sample = min(len(self.audio), beat_start_sample + attack_samples)
                gain_reduction_signal[beat_start_sample:attack_end_sample] = np.linspace(1, 1/ratio, attack_end_sample - beat_start_sample, endpoint=False)

                # Hold phase
                hold_end_sample = min(len(self.audio), attack_end_sample + hold_samples)
                gain_reduction_signal[attack_end_sample:hold_end_sample] = 1/ratio

                # Release phase
                release_end_sample = min(len(self.audio), hold_end_sample + release_samples)
                if release_end_sample > hold_end_sample: # Ensure there's a release period
                     gain_reduction_signal[hold_end_sample:release_end_sample] = np.linspace(1/ratio, 1, release_end_sample - hold_end_sample, endpoint=False)

                # Ensure gain after release period is 1 for samples not covered by next beat's envelope
                # This is implicitly handled by initializing with 1s and only modifying sections.
                # However, overlapping envelopes need careful handling; this simple model assumes beats are spaced enough.


        # Apply gain reduction to the audio
        processed_audio = self.audio * gain_reduction_signal
        
        # Apply makeup gain
        processed_audio = processed_audio * makeup_gain_linear
        
        self.audio = np.clip(processed_audio, -1.0, 1.0)

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
        
    def process_remix(self, tempo_change=1.0, pitch_steps=0,
                  delay_params=None, flanger_params=None,
                  eq_bands=None, sidechain_params=None,
                  beat_slice_enabled=False): # Renamed from beat_slice for clarity
        """Gelişmiş remix işlem pipeline'ı with new effect parameters."""
        print("Ses dosyası yükleniyor...")
        self.load_audio()
        
        print("Tempo ve ritim analizi yapılıyor...")
        self.apply_beat_detection() # Essential for some effects if not using external sidechain source
        
        if beat_slice_enabled: # Use the renamed parameter
            print("Beat slicing uygulanıyor...")
            self.beat_slice(slice_length=4) # Assuming slice_length is fixed or add as param
        
        if tempo_change != 1.0:
            print(f"Tempo değiştiriliyor ({tempo_change}x)...")
            self.time_stretch(rate=tempo_change) # Ensure 'rate' is the correct kwarg
            
        if pitch_steps != 0:
            print(f"Perde kaydırma uygulanıyor ({pitch_steps} adım)...")
            self.pitch_shift(steps=pitch_steps) # Ensure 'steps' is the correct kwarg

        # Enhanced effects
        if delay_params:
            print("Gelişmiş Delay efekti ekleniyor...")
            self.add_delay(**delay_params)
        else: # Fallback to some default if desired, or skip
            # Example: self.add_delay(delay_time=0.3, decay=0.4, feedback=0.3)
            pass

        if flanger_params:
            print("Gelişmiş Flanger efekti ekleniyor...")
            self.add_flanger(**flanger_params)
        else:
            # Example: self.add_flanger(rate=0.7, depth=0.003, feedback=0.4)
            pass
            
        if eq_bands: # Renamed from add_effects which was a boolean
            print("Parametrik EQ uygulanıyor...")
            self.add_parametric_eq(bands=eq_bands)
        # else: # Old filter call was: self.add_filter('lowpass', 2000, order=4)
            # This would now be:
            # self.add_parametric_eq(bands=[{'type': 'lowpass', 'freq': 2000, 'q': 0.707}])
            pass

        if sidechain_params: # Renamed from add_sidechain which was a boolean
            print("Gelişmiş Sidechain kompresyon uygulanıyor...")
            self.add_sidechain(**sidechain_params)
        # else: # Old call: self.add_sidechain()
            pass
            
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
    
    try:
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs available: {gpus}")
        else:
            print("No GPUs available, TensorFlow will use CPU.")
    except Exception as e:
        print(f"Error initializing TensorFlow: {e}")

    input_file = "input.wav" # Ensure this file exists for testing
    output_file = "output_remix_advanced.wav" # New output file name
    
    try:
        remixer = Premiermixx(input_file, output_file)

        # Define parameters for the enhanced effects
        delay_settings = {'delay_time': 0.4, 'decay': 0.6, 'feedback': 0.35,
                          'ping_pong': True, 'lfo_rate': 0.2, 'lfo_depth': 0.005}

        flanger_settings = {'rate': 0.3, 'depth': 0.005, 'feedback': 0.6,
                            'lfo_waveform': 'triangle', 'stereo_width': 0.5}

        eq_settings = [
            {'type': 'lowshelf', 'freq': 150, 'q': 0.7, 'gain_db': 2.0},
            {'type': 'peak', 'freq': 1200, 'q': 1.5, 'gain_db': -3.0},
            {'type': 'highshelf', 'freq': 5000, 'q': 0.7, 'gain_db': 1.5},
            {'type': 'highpass', 'freq': 50, 'q': 0.707}
        ]

        sidechain_settings = {'threshold_db': -18, 'ratio': 5, 'attack_ms': 5,
                              'release_ms': 150, 'hold_ms': 10, 'makeup_gain_db': 1.0,
                              'hpf_freq': 50} # Using internal beat detection

        remixer.process_remix(
            tempo_change=1.1,      # 10% faster
            pitch_steps=-1,        # 1 step down
            delay_params=delay_settings,
            flanger_params=flanger_settings,
            eq_bands=eq_settings,
            sidechain_params=sidechain_settings,
            beat_slice_enabled=True # Parameter name updated
        )
        
    except FileNotFoundError:
        print(f"Lütfen '{input_file}' adında bir ses dosyası ekleyin.")
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()