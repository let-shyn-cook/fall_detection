import cv2
import mediapipe as mp
import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load the trained model
with open('test_again.pkl', 'rb') as file:
    model = pickle.load(file)

columns = ('x1', 'y1', 'z1', 'v1',
           'x2', 'y2', 'z2', 'v2',
           'x3', 'y3', 'z3', 'v3',
           'x4', 'y4', 'z4', 'v4',
           'x5', 'y5', 'z5', 'v5',
           'x6', 'y6', 'z6', 'v6',
           'x7', 'y7', 'z7', 'v7',
           'x8', 'y8', 'z8', 'v8',
           'x9', 'y9', 'z9', 'v9',
           'x10', 'y10', 'z10', 'v10',
           'x11', 'y11', 'z11', 'v11',
           'x12', 'y12', 'z12', 'v12',
           'x13', 'y13', 'z13', 'v13',
           'x14', 'y14', 'z14', 'v14',
           'x15', 'y15', 'z15', 'v15',
           'x16', 'y16', 'z16', 'v16',
           'x17', 'y17', 'z17', 'v17',
           'x18', 'y18', 'z18', 'v18',
           'x19', 'y19', 'z19', 'v19',
           'x20', 'y20', 'z20', 'v20',
           'x21', 'y21', 'z21', 'v21',
           'x22', 'y22', 'z22', 'v22',
           'x23', 'y23', 'z23', 'v23',
           'x24', 'y24', 'z24', 'v24',
           'x25', 'y25', 'z25', 'v25',
           'x26', 'y26', 'z26', 'v26',
           'x27', 'y27', 'z27', 'v27',
           'x28', 'y28', 'z28', 'v28',
           'x29', 'y29', 'z29', 'v29',
           'x30', 'y30', 'z30', 'v30',
           'x31', 'y31', 'z31', 'v31',
           'x32', 'y32', 'z32', 'v32',
           'x33', 'y33', 'z33', 'v33')

cap = cv2.VideoCapture(2)
cap.set(3, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 * 2)
ret, frame = cap.read()
width, height = frame.shape[0], frame.shape[1]

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.5, model_complexity=2) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Export coordinates
        try:
            body_pose = results.pose_landmarks.landmark
            num_coords = len(body_pose)

            # Extract pose landmarks
            pose_row = list(np.array([[landmarks.x, landmarks.y, landmarks.z, landmarks.visibility] for landmarks in body_pose]).flatten())

            # Make predictions
            x = pd.DataFrame([pose_row], columns=columns)
            body_language_class = model.predict(x)[0]
            body_language_prob = model.predict_proba(x)[0]

            max_act = round(body_language_prob[np.argmax(body_language_prob)], 2)
            if max_act > 0.4:
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # Draw bounding box and status
            if body_pose:
                x_coords = [landmarks.x for landmarks in body_pose]
                y_coords = [landmarks.y for landmarks in body_pose]
                x_min = int(min(x_coords) * frame_width)
                x_max = int(max(x_coords) * frame_width)
                y_min = int(min(y_coords) * frame_height)
                y_max = int(max(y_coords) * frame_height)

                if body_language_class == 'fall' and max_act > 0.7:
                    color = (0, 0, 255)  # Red for fall
                    status = "Fall"
                else:
                    color = (0, 255, 0)  # Green for normal
                    status = "Normal"

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(image, status, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        except Exception as e:
            print(e)
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Fall Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
