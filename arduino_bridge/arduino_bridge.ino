// DrowsyGuard AI - Arduino graduated alarm + vibration
// Serial commands from Python:
//   '0' safe      -> green LED, motor run, silent
//   '1' low risk  -> yellow LED, soft beep 1x/2s, vibe 30%
//   '3' distracted-> yellow blink, double-beep, vibe 60%
//   '2' DROWSY    -> red LED, loud alarm loop, vibe 100%, motor STOP
//   'N' no face   -> yellow slow blink, silent, motor run
// Plus HC-SR04 < 30cm -> treated as '2' (auto-brake)
// Pins: buzzer 8, ledR 7, ledG 6, ledY 5, trig 9, echo 10,
//       motor L298N: ENA 3, IN1 4, IN2 2, vibration motor (PWM) 11

const int BUZZ = 8, LEDR = 7, LEDG = 6, LEDY = 5;
const int TRIG = 9, ECHO = 10;
const int ENA = 3, IN1 = 4, IN2 = 2, VIB = 11;
const long SAFE_CM = 30;

char cmd = '0';
unsigned long lastBeep = 0;
int beepStep = 0;

long readCm() {
  digitalWrite(TRIG, LOW); delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  return pulseIn(ECHO, HIGH, 30000) / 58;
}

void motorRun(bool run) {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  analogWrite(ENA, run ? 180 : 0);
}

void allLedsOff() {
  digitalWrite(LEDR, LOW); digitalWrite(LEDG, LOW); digitalWrite(LEDY, LOW);
}

void setup() {
  pinMode(BUZZ, OUTPUT); pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT); pinMode(LEDY, OUTPUT);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(VIB, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) { cmd = (char)Serial.read(); beepStep = 0; }
  long cm = readCm();
  char level = (cm > 0 && cm < SAFE_CM) ? '2' : cmd;  // obstacle = max danger
  unsigned long now = millis();
  allLedsOff();

  if (level == '2') {                    // DROWSY: loud + vibe max + brake
    digitalWrite(LEDR, HIGH);
    analogWrite(VIB, 255); motorRun(false);
    if (now - lastBeep > 350) {          // urgent 880Hz-ish fast loop
      lastBeep = now;
      digitalWrite(BUZZ, beepStep % 2 == 0 ? HIGH : LOW);
      beepStep++;
    }
  } else if (level == '3') {             // DISTRACTED: double-beep + vibe 60%
    digitalWrite(LEDY, HIGH);
    analogWrite(VIB, 150); motorRun(true);
    unsigned long t = (now / 250) % 8;   // beep-beep ..... pause
    digitalWrite(BUZZ, (t == 0 || t == 2) ? HIGH : LOW);
  } else if (level == '1') {             // LOW: soft tick + vibe 30%
    digitalWrite(LEDY, HIGH);
    analogWrite(VIB, 80); motorRun(true);
    if (now - lastBeep > 2000) { lastBeep = now; digitalWrite(BUZZ, HIGH); delay(120); digitalWrite(BUZZ, LOW); }
    else digitalWrite(BUZZ, LOW);
  } else if (level == 'N') {             // no face: slow blink, silent
    digitalWrite(LEDY, (now / 800) % 2);
    digitalWrite(BUZZ, LOW); analogWrite(VIB, 0); motorRun(true);
  } else {                               // safe
    digitalWrite(LEDG, HIGH);
    digitalWrite(BUZZ, LOW); analogWrite(VIB, 0); motorRun(true);
  }
}
