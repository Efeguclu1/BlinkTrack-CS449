import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe FaceMesh göz landmark indexleri
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(eye_points):
    vertical_1 = euclidean_dist(eye_points[1], eye_points[5])
    vertical_2 = euclidean_dist(eye_points[2], eye_points[4])
    horizontal = euclidean_dist(eye_points[0], eye_points[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def main():
    cap = cv2.VideoCapture(0)

    
    EAR_THRESHOLD = 0.87      # normalde ~0.93, kapalıda ~0.75 →  iyi  orta nokta
    FRAMES_LIMIT = 2          # en az 2 frame kapalıysa blink say
    closed_frames = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                face = results.multi_face_landmarks[0]

                left_eye = []
                right_eye = []

                for idx, lm in enumerate(face.landmark):
                    if idx in LEFT_EYE:
                        left_eye.append(np.array([lm.x * w, lm.y * h]))
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
                    if idx in RIGHT_EYE:
                        right_eye.append(np.array([lm.x * w, lm.y * h]))
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

                if len(left_eye) == 6 and len(right_eye) == 6:
                    left_ear = eye_aspect_ratio(np.array(left_eye))
                    right_ear = eye_aspect_ratio(np.array(right_eye))
                    ear = (left_ear + right_ear) / 2.0

                    # Ekrana EAR yaz
                    cv2.putText(frame, f"EAR: {ear:.3f}", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Blink logic
                    if ear < EAR_THRESHOLD:
                        closed_frames += 1
                    else:
                        if closed_frames >= FRAMES_LIMIT:
                            print("Blink detected!")
                        closed_frames = 0

            cv2.imshow("Blink Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
