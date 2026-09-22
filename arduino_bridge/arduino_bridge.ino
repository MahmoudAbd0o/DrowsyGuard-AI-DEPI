// Arduino bridge for Driver Drowsiness project (Uno)
// Receives from Python over Serial:
//   '1' = drowsy -> buzzer + red LED + stop motor
//   '0' = safe   -> green LED, motor run
//   'N' = no face-> yellow blink, keep motor
// Plus: HC-SR04 front distance -> auto-brake under 30cm
// Pins: buzzer 8, ledR 7, ledG 6, ledY 5, trig 9, echo 10,
//       motor (L298N): ENA 3, IN1 4, IN2 2

const int BUZZ = 8, LEDR = 7, LEDG = 6, LEDY = 5;
const int TRIG = 9, ECHO = 10;
const int ENA = 3, IN1 = 4, IN2 = 2;
const long SAFE_CM = 30;

char cmd = '0';

long readCm() {
  digitalWrite(TRIG, LOW); delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long d = pulseIn(ECHO, HIGH, 30000); // timeout 30ms
  return d / 58; // cm (0 if timeout)
}

void motorRun(bool run) {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  analogWrite(ENA, run ? 180 : 0);
}

void setup() {
  pinMode(BUZZ, OUTPUT); pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT); pinMode(LEDY, OUTPUT);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) cmd = (char)Serial.read();
  long cm = readCm();
  bool tooClose = (cm > 0 && cm < SAFE_CM);

  if (cmd == '1' || tooClose) {
    // DROWSY or obstacle: alarm + brake
    digitalWrite(LEDR, HIGH); digitalWrite(LEDG, LOW); digitalWrite(LEDY, LOW);
    digitalWrite(BUZZ, HIGH); motorRun(false);
    delay(300); digitalWrite(BUZZ, LOW); delay(200);
  } else if (cmd == 'N') {
    digitalWrite(LEDY, HIGH); digitalWrite(LEDR, LOW); digitalWrite(LEDG, LOW);
    digitalWrite(BUZZ, LOW); motorRun(true);
  } else {
    digitalWrite(LEDG, HIGH); digitalWrite(LEDR, LOW); digitalWrite(LEDY, LOW);
    digitalWrite(BUZZ, LOW); motorRun(true);
  }
}
