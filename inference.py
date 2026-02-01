import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# config
MODEL_PATH = 'model.keras'
THRESHOLD = 0.80
LABELS = np.array(['bad', 'book', 'bye', 'drink', 'hello', 'no', 'please', 'yes'])

# dashboard colors
COLORS = [
    (245, 117, 16), (117, 245, 16), (16, 117, 245), (200, 100, 200),
    (0, 255, 255), (0, 0, 255), (255, 255, 0), (255, 0, 255)
]

# setup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except IOError:
    print(f"Error: Could not find {MODEL_PATH}.")
    exit()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# mediapipe functions
def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    pose = np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([face, lh, pose, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {int(prob*100)}%", (0, 85 + num * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output_frame

# main loop
cap = cv2.VideoCapture(0)
sequence = []
sentence = []
predictions = []
res = np.zeros(len(LABELS))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    print("System ready. Press 'q' to quit.")

    # scale window
    cv2.namedWindow('ASL Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ASL Recognition', 1024, 768)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            pred_idx = np.argmax(res)
            predictions.append(pred_idx)
            
            if np.unique(predictions[-10:])[0] == pred_idx:
                if res[pred_idx] > THRESHOLD:
                    if len(sentence) > 0:
                        if LABELS[pred_idx] != sentence[-1]:
                            sentence.append(LABELS[pred_idx])
                    else:
                        sentence.append(LABELS[pred_idx])

            if len(sentence) > 5:
                sentence = sentence[-5:]

        image = cv2.flip(image, 1)
        image = prob_viz(res, LABELS, image, COLORS)
        
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('ASL Recognition', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            sentence = []

    cap.release()
    cv2.destroyAllWindows()