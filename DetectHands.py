import cv2
import mediapipe as mp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing...")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)
cv2.namedWindow('Detect hands by Cveramz', cv2.WINDOW_NORMAL)

message = "Press 'q' to quit."

def detectHandsLandmarks(frame, hands, display=False):
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    if display and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame, results

logger.info("Initializing camera...")

while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    frame, results = detectHandsLandmarks(frame, hands, display=True)
    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Fingers Counter', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or k == 27:
        logger.info("Exiting...")
        break

camera_video.release()
cv2.destroyAllWindows()
