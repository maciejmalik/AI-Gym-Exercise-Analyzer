import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

model_multi = tf.keras.models.load_model("model.h5")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(video_path, max_frames=237):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    valid_frames = 0
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            valid_frames+=1
            keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
        else:
            keypoints = np.zeros((33, 4))
        keypoints_list.append(keypoints.flatten())
    cap.release()
    while len(keypoints_list) < max_frames:
        keypoints_list.append(np.zeros((33 * 4,), dtype=np.float32))

    keypoints_array = np.array(keypoints_list)
    return keypoints_array, valid_frames

def predict(video_path):
    keypoints_array, valid_frames = extract_keypoints(video_path)
    total_frames = keypoints_array.shape[0]
    keypoints_array = np.expand_dims(keypoints_array, axis=0)
    MIN_VALID_RATIO = 0.3
    if total_frames == 0 or valid_frames / total_frames < MIN_VALID_RATIO:
        return (
            "Na nagraniu nie wykryto wyraźnej sylwetki ćwiczącego – nie można rozpoznać ćwiczenia.",
            None
                )
    preds = model_multi.predict(keypoints_array)
    print(preds)
    if isinstance(preds, dict):
        predictions_exercise = preds.get('exercise')
        predictions_form = preds.get('technique')
    else:
        predictions_exercise, predictions_form = preds[0], preds[1]

    probabilities = predictions_exercise[0]
    predicted_exercise_label = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    THRESHOLD = 0.5

    if confidence < THRESHOLD:
        return (
            "Nie rozpoznano żadnego ćwiczenia – model jest zbyt niepewny.",
            None
        )

    if predicted_exercise_label == 0:
        exercise_name = "Wykonywane ćwiczenie to wyciskanie na ławce poziomej,"
        exercise_image = "klata.png"
    elif predicted_exercise_label == 1:
        exercise_name = "Wykonywane ćwiczenie to pompka,"
        exercise_image = "klata.png"
    elif predicted_exercise_label == 2:
        exercise_name = "Wykonywane ćwiczenie to przysiad ze sztangą,"
        exercise_image = "Nogi.png"
    elif predicted_exercise_label == 3:
        exercise_name = "Wykonywane ćwiczenie to uginanie sztangi na biceps,"
        exercise_image = "Biceps.png"
    else:
        exercise_name = "Nieznane ćwiczenie,"
        exercise_image = None

    if predictions_form.ndim == 2:
        prob_form = predictions_form[0, 0]
    else:
        prob_form = predictions_form.ravel()[0]
    predicted_form_label = int(prob_form >= 0.5)
    exercise_form = "technika ćwiczenia jest poprawna" if predicted_form_label == 1 else "technika ćwiczenia jest niepoprawna"

    exercise = exercise_name + " a " + exercise_form
    return exercise, exercise_image

interface = gr.Interface(
    fn=predict,
    inputs=gr.Video(),
    outputs=[
        gr.Text(label="Opis"),
        gr.Image(label="Zaangażowane mięśnie", type="filepath")
    ],
    title="System oceny techniki ćwiczeń siłowych",
    description="Załaduj wideo, aby  rozpoznać ćwiczenie oraz ocenić technikę"
)

interface.launch()
