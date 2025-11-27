import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


SMOOTHING = 0.2
prev_x, prev_y = None, None

def smooth(value, prev):
    if prev is None:
        return value
    return prev + (value - prev) * SMOOTHING

def main():
    cap = cv2.VideoCapture(0)
    pyautogui.FAILSAFE = False

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        global prev_x, prev_y

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                face = results.multi_face_landmarks[0]

                # Nose tip (landmark index 1)
                lm = face.landmark[1]

                # Convert to screen coords
                screen_w, screen_h = pyautogui.size()
                target_x = int(screen_w * lm.x)
                target_y = int(screen_h * lm.y)

                # Smooth movement
                smoothed_x = smooth(target_x, prev_x)
                smoothed_y = smooth(target_y, prev_y)

                prev_x, prev_y = smoothed_x, smoothed_y

                # Move cursor
                pyautogui.moveTo(smoothed_x, smoothed_y, duration=0)

                # Visual debug
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

            cv2.imshow("Cursor Control Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
