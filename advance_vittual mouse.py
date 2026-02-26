import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Screen size
screen_w, screen_h = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

# Smoothening variables
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 7

frameR = 100  # Frame Reduction
click_delay = 0.3
last_click_time = 0

def fingers_up(handLms):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if handLms.landmark[4].x < handLms.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for tip in tips[1:]:
        if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    h, w, c = img.shape

    # Draw frame reduction box
    cv2.rectangle(img, (frameR, frameR), (w - frameR, h - frameR),
                  (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            lmList = []
            for id, lm in enumerate(handLms.landmark):
                lmList.append((int(lm.x * w), int(lm.y * h)))

            fingers = fingers_up(handLms)

            x1, y1 = lmList[8]   # Index
            x2, y2 = lmList[12]  # Middle

            # Move Mode (Only index finger up)
            if fingers[1] == 1 and fingers[2] == 0:

                # Convert coordinates
                screen_x = np.interp(x1, (frameR, w - frameR), (0, screen_w))
                screen_y = np.interp(y1, (frameR, h - frameR), (0, screen_h))

                # Smoothen movement
                clocX = plocX + (screen_x - plocX) / smoothening
                clocY = plocY + (screen_y - plocY) / smoothening

                pyautogui.moveTo(clocX, clocY)

                plocX, plocY = clocX, clocY

            # Left Click (Index + Middle up & close)
            if fingers[1] == 1 and fingers[2] == 1:
                distance = np.hypot(x2 - x1, y2 - y1)

                if distance < 30 and time.time() - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = time.time()

            # Right Click (Thumb + Index pinch)
            thumb_x, thumb_y = lmList[4]
            distance_thumb = np.hypot(thumb_x - x1, thumb_y - y1)

            if distance_thumb < 30 and time.time() - last_click_time > click_delay:
                pyautogui.rightClick()
                last_click_time = time.time()

            # Scroll Mode (Index + Middle up, far apart)
            if fingers[1] == 1 and fingers[2] == 1:
                if y2 < y1:
                    pyautogui.scroll(20)
                elif y2 > y1:
                    pyautogui.scroll(-20)

            # Drag Mode (Only Index up + hold pinch)
            if fingers[1] == 1 and distance_thumb < 25:
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()

    cv2.imshow("Advanced AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()