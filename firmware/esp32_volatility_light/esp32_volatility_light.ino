// Lampu status volatilitas BTCUSDT untuk ESP32 (ekstra, bukan bagian inti sistem).
//
// Polling GET <API_URL>/status tiap 30 detik, lalu:
//   LED hijau  : P(spike) < threshold/2, sistem OK
//   LED kuning : P(spike) >= threshold/2 (waspada)
//   LED merah  : P(spike) >= threshold (sama dengan kondisi alert Telegram), buzzer bunyi sekali
//   LED kuning berkedip: sistem STALE/DEGRADED atau API tidak terjangkau (jangan percaya angkanya)
// OLED SSD1306 128x64 menampilkan P(spike), threshold, status breaker, rezim.
//
// Library: ArduinoJson (v7), Adafruit SSD1306, Adafruit GFX.
// Simulasi: buka folder ini di Wokwi (diagram.json + wokwi.toml), WiFi "Wokwi-GUEST".

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";
const char* API_URL   = "https://crypto-vol-api.onrender.com";  // ganti dengan URL service kamu
const char* SYMBOL    = "BTCUSDT";
const uint32_t POLL_MS = 30000;

const int PIN_GREEN = 25, PIN_YELLOW = 26, PIN_RED = 27, PIN_BUZZER = 14;
Adafruit_SSD1306 oled(128, 64, &Wire, -1);

struct Reading {
  bool ok = false;
  float prob = NAN, threshold = NAN;
  String state = "?", regime = "", mode = "";
};

Reading fetchStatus() {
  Reading r;
  WiFiClientSecure client;
  client.setInsecure();  // demo: tanpa pinning sertifikat. Untuk produksi pasang root CA.
  HTTPClient http;
  http.setTimeout(15000);  // free tier bisa butuh waktu untuk bangun
  if (!http.begin(client, String(API_URL) + "/status")) return r;
  int code = http.GET();
  if (code == 200) {
    JsonDocument doc;
    if (!deserializeJson(doc, http.getStream())) {
      r.threshold = doc["threshold"] | NAN;
      r.state = String((const char*)(doc["circuit_breaker"]["state"] | "?"));
      JsonVariant latest = doc["latest"][SYMBOL];
      if (!latest.isNull()) {
        r.prob = latest["probability"] | NAN;
        r.regime = String((const char*)(latest["regime"] | ""));
        r.mode = String((const char*)(latest["mode"] | ""));
        r.ok = true;
      }
    }
  }
  http.end();
  return r;
}

void setLeds(bool g, bool y, bool red) {
  digitalWrite(PIN_GREEN, g);
  digitalWrite(PIN_YELLOW, y);
  digitalWrite(PIN_RED, red);
}

void beep() {
  for (int i = 0; i < 3; i++) { tone(PIN_BUZZER, 2000, 120); delay(200); }
  noTone(PIN_BUZZER);
}

void draw(const Reading& r) {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.printf("%s vol 15m", SYMBOL);
  oled.setTextSize(2);
  oled.setCursor(0, 14);
  if (r.ok && !isnan(r.prob)) oled.printf("%.1f%%", r.prob * 100); else oled.print("--");
  oled.setTextSize(1);
  oled.setCursor(0, 36);
  if (!isnan(r.threshold)) oled.printf("thr %.1f%%", r.threshold * 100);
  oled.setCursor(0, 46);
  oled.printf("sistem %s", r.state.c_str());
  oled.setCursor(0, 56);
  oled.print(r.regime);
  oled.display();
}

bool lastWasAlert = false;
Reading cur;

void setup() {
  Serial.begin(115200);
  pinMode(PIN_GREEN, OUTPUT); pinMode(PIN_YELLOW, OUTPUT); pinMode(PIN_RED, OUTPUT); pinMode(PIN_BUZZER, OUTPUT);
  oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE); oled.setCursor(0, 0); oled.print("WiFi..."); oled.display();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(250); Serial.print("."); }
  Serial.println(" terhubung");
}

uint32_t lastPoll = 0;

void loop() {
  if (lastPoll == 0 || millis() - lastPoll >= POLL_MS) {
    lastPoll = millis();
    cur = fetchStatus();
    Serial.printf("ok=%d prob=%.4f thr=%.4f state=%s\n", cur.ok, cur.prob, cur.threshold, cur.state.c_str());
    draw(cur);
  }
  bool healthy = cur.ok && cur.state == "OK" && !isnan(cur.prob) && !isnan(cur.threshold);
  if (!healthy) {  // data tidak bisa dipercaya: LED kuning berkedip, tidak ada buzzer
    bool on = (millis() / 500) % 2;
    setLeds(false, on, false);
    lastWasAlert = false;
  } else {
    bool alert = cur.prob >= cur.threshold;
    setLeds(!alert && cur.prob < cur.threshold / 2, !alert && cur.prob >= cur.threshold / 2, alert);
    if (alert && !lastWasAlert) beep();
    lastWasAlert = alert;
  }
  delay(50);
}
