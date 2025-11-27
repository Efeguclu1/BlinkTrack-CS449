import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def main():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        h, w, _ = frame.shape
                        x, y = int(lm.x * w), int(lm.y * h)

                        # Only draw eye keypoints 
                        if 133 < idx < 468:  # highlight facial points
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            cv2.imshow("Gaze Tracking - FaceMesh Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
