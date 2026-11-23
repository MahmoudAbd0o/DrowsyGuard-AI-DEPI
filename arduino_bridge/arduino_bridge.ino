// =====================================================================
// DrowsyGuard AI - car firmware
// L298N + steering servo + HC-SR04 + MPU6050 + 3 LEDs + buzzer + HC-06
//
// Drop-in replacement for the team's Bluetooth car sketch: same pins,
// same driving commands, plus the DrowsyGuard camera protocol.
//
// Both links are listened to at the same time (9600 baud):
//   * Bluetooth  HC-06  TX -> D10, RX -> D11 (SoftwareSerial)
//   * USB cable  Serial (for debugging / running without Bluetooth)
//
// CAMERA / AI  (sent by dashboard.py, heartbeat every ~1 s)
//   '0' safe        -> green LED
//   '1' low risk    -> RED led + soft beep (eyes closing, car still drives)
//   '3' distracted  -> white LED + faster beep
//   '2' DROWSY      -> red LED + loud buzzer + HARD BRAKE
//   'N' no face     -> white LED, silent (never reported as "safe")
//
// The grace period lives on the laptop (`DROWSY_CONFIRM_SEC`, 30 s): eyes
// closed ramp '1' -> '2' only when the closure is sustained, so a blink never
// stops the car.
//
// MANUAL DRIVING (phone app / keyboard)
//   'S' stop + ARM the motors   'F' forward   'B' back
//   'L' left                    'R' right
//
// SAFETY
//   * motors stay locked until the first 'S'
//   * '2' cancels an automatic turn and stops the car immediately
//   * no camera byte for CAMERA_TIMEOUT ms -> 'N' (unknown), not '0'
//   * HC-SR04 closer than STOP_CM -> obstacle alarm
// =====================================================================

#include <SoftwareSerial.h>
#include <Wire.h>
#include <Servo.h>
#include <math.h>

// =====================================
// BLUETOOTH (HC-06)
// =====================================

SoftwareSerial bt(10, 11);

// =====================================
// L298N
// =====================================

const byte in1 = 2;
const byte in2 = 3;
const byte in3 = 4;
const byte in4 = 5;

// =====================================
// ULTRASONIC
// =====================================

const byte TRIG = 6;
const byte ECHO = 7;

// =====================================
// SERVO
// =====================================

const byte SERVO_PIN = 9;

const int SERVO_CENTER = 90;
const int SERVO_LEFT = 150;
const int SERVO_RIGHT = 30;

// =====================================
// LEDS AND BUZZER
// =====================================

const byte GREEN_LED = A0;
const byte WHITE_LED = A1;
const byte RED_LED = A2;
const byte BUZZER = A3;

// =====================================
// MPU6050
// =====================================

const byte MPU = 0x68;

// =====================================
// SETTINGS
// =====================================

const int STOP_CM = 20;
const int SIDE_CLEAR_CM = 25;

const unsigned long COMMAND_TIMEOUT = 600;
const unsigned long CAMERA_TIMEOUT = 3000;   // dashboard heartbeat ~1 s

const unsigned long SERVO_WAIT = 400;
const unsigned long TURN_TIME = 500;
const unsigned long RETRY_TIME = 1500;

const unsigned long DISTANCE_INTERVAL = 70;

// =====================================
// OBJECTS
// =====================================

Servo myServo;

// =====================================
// SAFETY SYSTEM
// =====================================

bool movementArmed = false;

char want = 'S';
char cur = 'S';

char alertState = 'N';

bool scanning = false;
bool turning = false;
bool blockedHold = false;
bool blocked = false;

bool emergencyStop = false;

// =====================================
// DISTANCES
// =====================================

int frontDistance = 999;
int leftDistance = 999;
int rightDistance = 999;

int servoAngle = SERVO_CENTER;

char autoTurn = 'S';

byte closeCount = 0;
byte scanStep = 0;

// =====================================
// TIMERS
// =====================================

unsigned long lastCmd = 0;
unsigned long lastDist = 0;
unsigned long lastTilt = 0;
unsigned long lastAlertRx = 0;
unsigned long lastBeep = 0;

unsigned long scanTimer = 0;
unsigned long turnTimer = 0;
unsigned long retryTimer = 0;

bool beepState = false;

// =====================================
// MOTOR CONTROL
// =====================================

void setPins(byte a, byte b, byte c, byte d) {

  digitalWrite(in1, a);
  digitalWrite(in2, b);

  digitalWrite(in3, c);
  digitalWrite(in4, d);
}

void drive(char command) {

  switch (command) {

    case 'F':
      setPins(LOW, HIGH, HIGH, LOW);
      break;

    case 'B':
      setPins(HIGH, LOW, LOW, HIGH);
      break;

    case 'L':
      setPins(LOW, HIGH, LOW, HIGH);
      break;

    case 'R':
      setPins(HIGH, LOW, HIGH, LOW);
      break;

    default:
      setPins(LOW, LOW, LOW, LOW);
      break;
  }
}

void stopCar() {

  setPins(LOW, LOW, LOW, LOW);

  cur = 'S';
}

// =====================================
// SERVO
// =====================================

void setServoAngle(int angle) {

  angle = constrain(angle, 0, 180);

  myServo.write(angle);

  servoAngle = angle;
}

// =====================================
// ULTRASONIC
// =====================================

int readCm() {

  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG, LOW);

  unsigned long duration =
    pulseIn(ECHO, HIGH, 12000);

  if (duration == 0)
    return 999;

  return duration / 58;
}

// =====================================
// MPU6050
// =====================================

bool readMPU(int &pitch, int &roll) {

  Wire.beginTransmission(MPU);

  Wire.write(0x3B);

  if (Wire.endTransmission(false) != 0)
    return false;

  if (Wire.requestFrom((uint8_t)MPU, (uint8_t)6) != 6)
    return false;

  int16_t ax =
    ((int16_t)Wire.read() << 8) | Wire.read();

  int16_t ay =
    ((int16_t)Wire.read() << 8) | Wire.read();

  int16_t az =
    ((int16_t)Wire.read() << 8) | Wire.read();

  pitch = atan2(
    (float)ax,
    sqrt((float)ay * ay + (float)az * az)
  ) * 180.0 / PI;

  roll = atan2(
    (float)ay,
    sqrt((float)ax * ax + (float)az * az)
  ) * 180.0 / PI;

  return true;
}

// =====================================
// SEND MPU DATA
// =====================================

void sendTilt() {

  if (millis() - lastTilt < 300)
    return;

  lastTilt = millis();

  int pitch, roll;

  if (readMPU(pitch, roll)) {

    bt.print('T');
    bt.print(pitch);
    bt.print(',');
    bt.println(roll);
  }
}

// =====================================
// COMMAND PARSER (shared by both links)
// =====================================

void handleChar(char c) {

  // Ignore line endings and spaces

  if (c == '\n' || c == '\r' || c == ' ')
    return;

  // CAMERA / AI

  if (c == '0' || c == '1' ||
      c == '2' || c == '3' || c == 'N') {

    alertState = c;

    lastAlertRx = millis();

    Serial.print("CAMERA: ");
    Serial.println(c);

    return;
  }

  // STOP COMMAND
  // Also arms the movement system.

  if (c == 'S') {

    movementArmed = true;

    want = 'S';

    lastCmd = millis();

    cancelAutomatic();

    stopCar();

    Serial.println("STOP - SYSTEM ARMED");

    return;
  }

  // MOVEMENT COMMANDS

  if (c == 'F' || c == 'B' ||
      c == 'L' || c == 'R') {

    // Ignore movement before receiving S

    if (!movementArmed) {

      Serial.println("MOVEMENT LOCKED");

      return;
    }

    want = c;

    lastCmd = millis();

    Serial.print("MOVEMENT: ");
    Serial.println(c);
  }
}

// Read Bluetooth + USB. The camera may be on either one.

void readLinks() {

  while (bt.available())
    handleChar(bt.read());

  while (Serial.available())
    handleChar(Serial.read());
}

// =====================================
// CAMERA ALERTS
// =====================================

void updateAlert(bool obstacleAlarm) {

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(WHITE_LED, LOW);
  digitalWrite(RED_LED, LOW);

  if (obstacleAlarm) {

    digitalWrite(RED_LED, HIGH);

    tone(BUZZER, 2000);

    return;
  }

  if (alertState == '0') {

    digitalWrite(GREEN_LED, HIGH);

    noTone(BUZZER);
  }

  else if (alertState == '2') {

    digitalWrite(RED_LED, HIGH);

    tone(BUZZER, 2000);
  }

  else if (alertState == '1') {

    // Eyes closing / low risk: red warning LED, soft beep, NO brake yet.
    // The laptop turns this into '2' after DROWSY_CONFIRM_SEC (1.2 s).

    digitalWrite(RED_LED, HIGH);

    if (millis() - lastBeep >= 900) {

      lastBeep = millis();

      beepState = !beepState;

      if (beepState)
        tone(BUZZER, 1200);
      else
        noTone(BUZZER);
    }
  }

  else if (alertState == '3') {

    digitalWrite(WHITE_LED, HIGH);

    if (millis() - lastBeep >= 700) {

      lastBeep = millis();

      beepState = !beepState;

      if (beepState)
        tone(BUZZER, 1500);

      else
        noTone(BUZZER);
    }
  }

  else {

    digitalWrite(WHITE_LED, HIGH);

    noTone(BUZZER);
  }
}

// =====================================
// START SCAN
// =====================================

void startScan() {

  scanning = true;
  turning = false;

  scanStep = 1;

  stopCar();

  leftDistance = 999;
  rightDistance = 999;

  setServoAngle(SERVO_LEFT);

  scanTimer = millis();

  Serial.println("SCANNING STARTED");
}

// =====================================
// CANCEL AUTOMATIC
// =====================================

void cancelAutomatic() {

  scanning = false;
  turning = false;
  blockedHold = false;

  scanStep = 0;

  setServoAngle(SERVO_CENTER);

  stopCar();
}

// =====================================
// PROCESS SCAN
// =====================================

void processScan() {

  if (!scanning)
    return;

  if (scanStep == 1) {

    if (millis() - scanTimer >= SERVO_WAIT) {

      leftDistance = readCm();

      setServoAngle(SERVO_RIGHT);

      scanStep = 2;

      scanTimer = millis();
    }
  }

  else if (scanStep == 2) {

    if (millis() - scanTimer >= SERVO_WAIT) {

      rightDistance = readCm();

      setServoAngle(SERVO_CENTER);

      scanStep = 3;

      scanTimer = millis();
    }
  }

  else if (scanStep == 3) {

    if (millis() - scanTimer >= SERVO_WAIT) {

      scanning = false;

      bool leftFree =
        leftDistance >= SIDE_CLEAR_CM;

      bool rightFree =
        rightDistance >= SIDE_CLEAR_CM;

      if (!leftFree && !rightFree) {

        blockedHold = true;

        retryTimer = millis();

        stopCar();

        Serial.println("BOTH SIDES BLOCKED");

        return;
      }

      if (leftFree && rightFree) {

        autoTurn =
          (leftDistance >= rightDistance) ? 'L' : 'R';
      }

      else if (leftFree) {

        autoTurn = 'L';
      }

      else {

        autoTurn = 'R';
      }

      turning = true;

      turnTimer = millis();

      drive(autoTurn);

      cur = autoTurn;

      Serial.print("AUTO TURN: ");
      Serial.println(autoTurn);
    }
  }
}

// =====================================
// PROCESS TURN
// =====================================

void processTurn() {

  if (!turning)
    return;

  if (millis() - turnTimer >= TURN_TIME) {

    stopCar();

    turning = false;

    blocked = false;

    closeCount = 0;

    blockedHold = false;

    lastDist = millis();

    Serial.println("TURN FINISHED");
  }
}

// =====================================
// SETUP
// =====================================

void setup() {

  // Initialize motor outputs FIRST

  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  stopCar();

  // Serial (USB) + Bluetooth

  Serial.begin(9600);

  bt.begin(9600);

  // Ultrasonic

  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);

  digitalWrite(TRIG, LOW);

  // LEDs

  pinMode(GREEN_LED, OUTPUT);
  pinMode(WHITE_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  pinMode(BUZZER, OUTPUT);

  // Initial safe state

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(WHITE_LED, LOW);
  digitalWrite(RED_LED, LOW);

  noTone(BUZZER);

  // Servo

  myServo.attach(SERVO_PIN);

  setServoAngle(SERVO_CENTER);

  // MPU6050

  Wire.begin();

#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(3000, true);
#endif

  Wire.beginTransmission(MPU);

  Wire.write(0x6B);
  Wire.write(0);

  Wire.endTransmission(true);

  // Safety lock

  movementArmed = false;

  want = 'S';

  cur = 'S';

  alertState = 'N';

  scanning = false;
  turning = false;
  blockedHold = false;

  lastCmd = millis();
  lastAlertRx = millis();
  lastDist = millis();

  Serial.println("==========================");

  Serial.println("DROWSYGUARD CAR READY");

  Serial.println("MOTOR SAFETY LOCK: ON");

  Serial.println("SEND S TO ARM");

  Serial.println("WAITING FOR CAMERA / REMOTE");

  Serial.println("==========================");
}

// =====================================
// MAIN LOOP
// =====================================

void loop() {

  // 1. Bluetooth + USB

  readLinks();

  // 2. Camera timeout -> unknown, never "safe"

  if (millis() - lastAlertRx > CAMERA_TIMEOUT) {

    alertState = 'N';
  }

  // 3. DROWSY = hard brake + loud alarm

  if (alertState == '2') {

    if (!emergencyStop) {

      emergencyStop = true;

      cancelAutomatic();
    }

    stopCar();

    updateAlert(false);

    sendTilt();

    return;
  }

  emergencyStop = false;

  // 4. Movement timeout

  if (movementArmed &&
      millis() - lastCmd > COMMAND_TIMEOUT) {

    want = 'S';

    stopCar();
  }

  // 5. Do not move before arming

  if (!movementArmed) {

    stopCar();

    updateAlert(false);

    sendTilt();

    return;
  }

  // 6. Manual command overrides automatic

  if (want != 'F' &&
      (scanning || turning || blockedHold)) {

    cancelAutomatic();
  }

  // 7. Front distance

  if (!scanning && !turning &&
      millis() - lastDist >= DISTANCE_INTERVAL) {

    lastDist = millis();

    frontDistance = readCm();

    if (frontDistance <= STOP_CM) {

      if (closeCount < 2)
        closeCount++;
    }

    else {

      closeCount = 0;
    }

    blocked = (closeCount >= 2);
  }

  // 8. Scanning

  if (scanning) {

    processScan();

    updateAlert(true);

    sendTilt();

    return;
  }

  // 9. Automatic turning

  if (turning) {

    processTurn();

    if (turning) {

      updateAlert(true);

      sendTilt();

      return;
    }
  }

  // 10. Both sides blocked

  if (blockedHold && want == 'F') {

    stopCar();

    if (millis() - retryTimer >= RETRY_TIME) {

      retryTimer = millis();

      startScan();
    }

    updateAlert(true);

    sendTilt();

    return;
  }

  // 11. Front obstacle

  if (want == 'F' && blocked) {

    startScan();

    updateAlert(true);

    sendTilt();

    return;
  }

  // 12. Normal movement

  if (want != cur) {

    drive(want);

    cur = want;
  }

  // 13. Alerts

  updateAlert(false);

  // 14. MPU

  sendTilt();
}
