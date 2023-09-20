import cv2 as cv
import numpy as np
import mediapipe as mp

cap = cv.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh

# Iris indices
LEFT_IRIS = list(range(474, 478))  # Indices for the left iris
RIGHT_IRIS = list(range(469, 473))  # Indices for the right iris

with mp_face_mesh.FaceMesh(
    max_num_faces=10,  # Detect 10 faces, can be adjusted. Larger number degrades performance
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the image horizontally like a mirror
        frame = cv.flip(frame, 1)

        # Recolor the image to RGB
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Width and height of the image
        height, width = frame.shape[:2]

        # Process the image and find faces
        results = face_mesh.process(rgb)

        # Extract landmarks for Iris Detection
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in landmarks.landmark])

                # Get center and radius of the left and right irises
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                # Convert all values to int
                l_center = np.array([int(l_cx), int(l_cy)], dtype=np.int32)
                r_center = np.array([int(r_cx), int(r_cy)], dtype=np.int32)

                # Draw circles on Irises
                if l_radius > 0:
                    cv.circle(frame, l_center, int(l_radius), (0, 255, 0), 1, cv.LINE_AA)
                if r_radius > 0:
                    cv.circle(frame, r_center, int(r_radius), (0, 255, 0), 1, cv.LINE_AA)

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv.imshow('Frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
