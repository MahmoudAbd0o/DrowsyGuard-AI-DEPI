"""Quick camera sanity check. Run with drowsiness env."""
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not found. Check privacy settings / cable.")
    raise SystemExit(1)

print("Camera OK. Press Q to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("ERROR: Failed to read frame.")
        break
    frame = cv2.flip(frame, 1)  # mirror view
    cv2.putText(frame, "Camera OK - press Q", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("test_camera", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
