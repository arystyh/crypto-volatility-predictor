# Firmware ekstra: lampu status ESP32

Bukan bagian inti sistem. ESP32 membaca `GET /status` tiap 30 detik dan menyalakan LED hijau,
kuning, atau merah sesuai P(spike) terhadap threshold yang sedang berlaku, plus buzzer saat
melewati threshold. Saat breaker STALE/DEGRADED atau API tidak terjangkau, LED kuning berkedip
dan buzzer diam, karena angkanya tidak bisa dipercaya.

- `esp32_volatility_light.ino`: sketch Arduino (ArduinoJson v7, Adafruit SSD1306/GFX).
- `diagram.json`, `libraries.txt`, `wokwi.toml`: simulasi Wokwi. Paling mudah: buat proyek
  ESP32 baru di wokwi.com, tempel isi `.ino`, `diagram.json`, dan `libraries.txt`.

Ganti `API_URL` dengan URL service kamu. Sketch memakai `setInsecure()` (tanpa verifikasi
sertifikat) supaya demo sederhana; untuk perangkat yang dipasang permanen, pasang root CA.
