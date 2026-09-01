import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
import time

# 1. Download the required modern MediaPipe model file if it doesn't exist
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading MediaPipe hand tracking model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Initialize the modern MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    min_hand_detection_confidence=0.8,
    running_mode=VisionRunningMode.VIDEO
)
landmarker = HandLandmarker.create_from_options(options)

# 3. Setup Webcam and Canvas
cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0
draw_color = (255, 0, 255)

print("Air Canvas is ready! Draw with your index finger. Press 'c' to clear, 'q' to quit.")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert frame to MediaPipe Image format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Calculate timestamp in milliseconds for the continuous video stream
    timestamp_ms = int((time.time() - start_time) * 1000)

    # Process the frame
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # 1. Stricter Gesture Detection
            # Compare the fingertip (8, 12, 16, 20) to the BASE knuckle (5, 9, 13, 17)
            # instead of the middle joint. This requires the finger to be fully extended.
            index_up = hand_landmarks[8].y < hand_landmarks[5].y
            middle_up = hand_landmarks[12].y < hand_landmarks[9].y
            ring_up = hand_landmarks[16].y < hand_landmarks[13].y
            pinky_up = hand_landmarks[20].y < hand_landmarks[17].y
            
            fingers_count = sum([index_up, middle_up, ring_up, pinky_up])
            x, y = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)

            # 2. Prevent "Leaving Strokes" (Anti-Glitch)
            # If the AI loses tracking and your finger jumps across the screen, break the line
            if prev_x != 0 and prev_y != 0:
                jump_dist = ((x - prev_x)**2 + (y - prev_y)**2) ** 0.5
                if jump_dist > 150:  # If distance is greater than 150 pixels
                    prev_x, prev_y = 0, 0

            # 3. Gesture Logic
            # STRICT DRAW: Index finger MUST be up, and middle/ring MUST be down
            if index_up and not middle_up and not ring_up:
                current_color = (255, 0, 255)
                brush_size = 5
                cv2.circle(frame, (x, y), 8, current_color, cv2.FILLED)
                
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                    
                cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, brush_size)
                prev_x, prev_y = x, y

            # STRICT ERASE: At least 3 fingers fully extended
            elif fingers_count >= 3:
                # Draw directly with a circle instead of a line for smoother erasing
                cv2.circle(canvas, (x, y), 50, (0, 0, 0), cv2.FILLED) 
                
                # Visual UI for the eraser
                cv2.circle(frame, (x, y), 50, (200, 200, 200), 2) 
                prev_x, prev_y = 0, 0 # Keep pen lifted while erasing
                
            else:
                # HOVER MODE (Fist, peace sign, etc.)
                prev_x, prev_y = 0, 0
    else:
        # Reset coordinates if hand goes out of frame
        prev_x, prev_y = 0, 0

    # Merge Canvas with Live Camera
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY_INV)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
    final_frame = cv2.add(frame_bg, canvas)

    cv2.imshow("AI Air Drawing", final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()