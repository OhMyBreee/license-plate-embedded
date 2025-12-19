#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ArduinoJson.h>

// =====================
// WIFI CONFIG
// =====================
const char* ssid = "laptop hotspot";
const char* password = "nasiayam123";

// =====================
// BACKEND
// =====================
const char* statusUrl = "http://192.168.137.1:8000/status";

// =====================
// I2C LCD (16x2)
// =====================
LiquidCrystal_I2C lcd(0x27, 16, 2);

// =====================
// LED PINS
// =====================
#define LED_WIFI   18
#define LED_MATCH  19

// =====================
// TIMING
// =====================
unsigned long lastBlink = 0;
bool wifiLedState = false;
unsigned long lastPoll = 0;

// =====================
// SETUP
// =====================
void setup() {
  Serial.begin(115200);

  pinMode(LED_WIFI, OUTPUT);
  pinMode(LED_MATCH, OUTPUT);

  digitalWrite(LED_WIFI, LOW);
  digitalWrite(LED_MATCH, LOW);

  // I2C
  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("Starting...");
  
  WiFi.begin(ssid, password);
}

// =====================
// WIFI LED HANDLER
// =====================
void handleWifiLed() {
  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(LED_WIFI, HIGH); // solid ON
  } else {
    unsigned long now = millis();
    if (now - lastBlink >= 500) {
      lastBlink = now;
      wifiLedState = !wifiLedState;
      digitalWrite(LED_WIFI, wifiLedState);
    }
  }
}

// =====================
// FETCH STATUS FROM API
// =====================
void fetchStatus() {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(statusUrl);
  http.setTimeout(3000);

  int code = http.GET();
  if (code != 200) {
    Serial.println("Failed to fetch status");
    http.end();
    return;
  }

  String payload = http.getString();
  http.end();

  Serial.println(payload);

  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    Serial.println("JSON error");
    return;
  }

  String plate = doc["plate"] | "---";
  bool match = doc["match"] | false;

  // =====================
  // LCD UPDATE
  // =====================
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("PLATE:");
  lcd.setCursor(7,0);
  lcd.print(plate);
  lcd.setCursor(0, 1);
  lcd.print("MATCH:");
  lcd.setCursor(7, 1);
  lcd.print(match ? "YES" : "NO");

  delay(3000);

  lcd.clear();

  // =====================
  // MATCH LED
  // =====================
  digitalWrite(LED_MATCH, match ? HIGH : LOW);
}

// =====================
// LOOP
// =====================
void loop() {
  handleWifiLed();

  if (WiFi.status() == WL_CONNECTED) {
    unsigned long now = millis();
    if (now - lastPoll >= 3000) {
      lastPoll = now;
      fetchStatus();
    }
  }
}
