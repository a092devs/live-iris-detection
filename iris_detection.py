import cv2 as cv
import numpy as np
import mediapipe as mp

cap = cv.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh

# Left iris indices
LEFT_IRIS = [474, 475, 476, 477]
# Right iris indices
RIGHT_IRIS = [469, 470, 471, 472]

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # important for iris detection, as it has 478 landmarks instead of 468
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

        # Extract landmarks
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points = np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark]) # detects 478 landmarks
            # print(mesh_points.shape)

            # Draw lines on the boundary of eyelids
            # cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)

            # Draw squares on iris
            # cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0, 255, 0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0, 255, 0), 1, cv.LINE_AA)

            # Get center and radius of the iris
            # l_cx = left iris center x, l_cy = left iris center y, l_radius = left iris radius
            # r_cx = right iris center x, r_cy = right iris center y, r_radius = right iris radius
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            # Convert all values to int
            center_left = np.array([int(l_cx), int(l_cy)], dtype=np.int32)
            center_right = np.array([int(r_cx), int(r_cy)], dtype=np.int32)

            # Draw circles on iris
            cv.circle(frame, center_left, int(l_radius), (0, 255, 0), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (0, 255, 0), 1, cv.LINE_AA)

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv.imshow('Frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
