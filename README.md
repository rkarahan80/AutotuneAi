# 🎵 Premiermixx

Premiermixx, yapay zeka destekli açık kaynaklı bir müzik remix uygulamasıdır. Müzik parçalarınızı profesyonel düzeyde remixlemenize, efektler eklemenize ve detaylı analizler yapmanıza olanak sağlar.

## ✨ Özellikler

- 🎚️ Tempo değiştirme ve beat senkronizasyonu
- 🎛️ Pitch shifting (perde kaydırma)
- ✂️ Beat slicing ve yeniden düzenleme
- 🔊 Profesyonel efektler:
  - Gelişmiş Delay (feedback kontrolü)
  - Gelişmiş Flanger
  - 🌫️ Reverb Efekti (Decay, Damping, Mix kontrollü)
  - 🎚️ Parametrik EQ (Lowpass, Highpass, Bandpass, Bandstop filtreleri, Frekans, Q ve Order kontrollü)
  - Sidechain Kompresyon
- 🔄 Crossfade destekli loop oluşturma
- 📊 Beat analizi ve görselleştirme
- 📈 Detaylı remix analizi
- 🖥️ Kullanıcı dostu grafik arayüz

## 🚀 Kurulum

1. Repoyu klonlayın:
```bash
git clone https://github.com/yourusername/Premiermixx.git
cd Premiermixx
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

## 💻 Kullanım

Grafik arayüzü başlatmak için:
```bash
python gui.py
```

veya komut satırı arayüzü için:
```bash
python main.py
```

### Grafik Arayüz Özellikleri

- Sürükle-bırak dosya desteği
- Gerçek zamanlı parametre kontrolü
- Görsel analiz gösterimi
- İlerleme takibi
- Kolay efekt yönetimi

### Parametreler

```python
remixer = Premiermixx("input.wav", "output_remix.wav")
remixer.process_remix(
    tempo_change=1.0,      # Tempo çarpanı (1.0 = normal)
    pitch_steps=0,         # Perde kaydırma adımı (yarım ton cinsinden)
    add_effects=True,      # Ana efektleri (Delay, Flanger) aktif/pasif yapar
    beat_slice=False,      # Beat slicing özelliğini aktif/pasif yapar
    add_sidechain=False,   # Sidechain kompresyonu aktif/pasif yapar

    # Reverb Efekti Parametreleri
    reverb_decay_time=0.5, # Reverb süresi (saniye, 0: kapalı)
    reverb_damping=0.5,    # Reverb sönümlemesi (0-1, yüksek frekansları etkiler)
    reverb_mix=0.25,       # Reverb ıslak/kuru karışımı (0-1, 0: kapalı)

    # Parametrik EQ Filtre Parametreleri
    eq_filter_type='bandpass', # Filtre tipi ('lowpass', 'highpass', 'bandpass', 'bandstop', veya None)
    eq_center_freq=1000,   # Merkez/Kesim frekansı (Hz)
    eq_q_factor=1.0,       # Q faktörü (bandpass/bandstop için geçerli)
    eq_order=4             # Filtre derecesi/eğimi
)
```

## 📊 Çıktılar

- `output_remix.wav`: Remixlenmiş ses dosyası
- `remix_analysis.png`: Gelişmiş remix analiz grafiği:
  - Dalga formu
  - Spektrogram
  - Beat tracking görselleştirmesi

## 🤝 Katkıda Bulunma

1. Bu repoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: X'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak kütüphaneleri kullanmaktadır:
- librosa
- numpy
- scipy
- pyworld
- soundfile
- matplotlib
- PyQt6