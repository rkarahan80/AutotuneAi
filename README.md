# 🎵 Premiermixx

Premiermixx, yapay zeka destekli açık kaynaklı bir müzik remix uygulamasıdır. Müzik parçalarınızı profesyonel düzeyde remixlemenize, efektler eklemenize ve detaylı analizler yapmanıza olanak sağlar.

## ✨ Özellikler

- 🎚️ Tempo değiştirme ve beat senkronizasyonu
- 🎛️ Pitch shifting (perde kaydırma)
- ✂️ Beat slicing ve yeniden düzenleme
- 🔊 Profesyonel efektler:
  - Gelişmiş Delay (feedback kontrolü)
  - Gelişmiş Flanger
  - Parametrik Filtreler
  - Sidechain Kompresyon
- 🔄 Crossfade destekli loop oluşturma
- 📊 Beat analizi ve görselleştirme
- 📈 Detaylı remix analizi

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

1. Remixlemek istediğiniz ses dosyasını `input.wav` olarak kaydedin
2. Programı çalıştırın:
```bash
python main.py
```

### Parametreler

```python
remixer = Premiermixx("input.wav", "output_remix.wav")
remixer.process_remix(
    tempo_change=1.2,      # Tempo çarpanı (1.0 = normal)
    pitch_steps=2,         # Perde kaydırma adımı
    add_effects=True,      # Efektleri aktif/pasif yapar
    beat_slice=True,       # Beat slicing özelliğini aktif/pasif yapar
    add_sidechain=True     # Sidechain kompresyonu aktif/pasif yapar
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