# 🎵 AutotuneAI

AutotuneAI, yapay zeka destekli açık kaynaklı bir ses işleme ve autotune uygulamasıdır. Ses kayıtlarınızı otomatik olarak düzeltir, efektler ekler ve detaylı analizler sunar.

## ✨ Özellikler

- 🎯 Otomatik pitch düzeltme (Autotune)
- 🔊 Ses efektleri (Eko, Reverb)
- ⚡ Hız ayarlama
- 📊 Detaylı ses analizi
- 📈 Görsel grafikler (Dalga formu ve Spektrogram)

## 🚀 Kurulum

1. Repoyu klonlayın:
```bash
git clone https://github.com/yourusername/AutotuneAI.git
cd AutotuneAI
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

## 💻 Kullanım

1. İşlemek istediğiniz ses dosyasını `input.wav` olarak kaydedin
2. Programı çalıştırın:
```bash
python main.py
```

### Parametreler

```python
processor = AudioProcessor("input.wav", "output.wav")
processor.process(
    add_effects=True,     # Efektleri aktif/pasif yapar
    speed_factor=1.0,     # Ses hızı (1.0 = normal)
    analyze=True,         # Ses analizi yapar
    visualize=True        # Grafikleri oluşturur
)
```

## 📊 Çıktılar

- `output_processed.wav`: İşlenmiş ses dosyası
- `waveform.png`: Ses dalgası grafiği
- `spectrogram.png`: Spektrogram görüntüsü
- Detaylı ses analizi sonuçları

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