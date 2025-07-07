import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

model_form = tf.keras.models.load_model("model.h5")
model_exercise = tf.keras.models.load_model("model2.h5")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def extract_keypoints(video_path, max_frames=207):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
        else:
            keypoints = np.zeros((33, 4))
        keypoints_list.append(keypoints.flatten())

    cap.release()

    while len(keypoints_list) < max_frames:
        keypoints_list.append(np.zeros((33 * 4,), dtype=np.float32))

    keypoints_array = np.array(keypoints_list)
    return keypoints_array
def predict(video_path):
    keypoints_array = extract_keypoints(video_path)

    keypoints_array = np.expand_dims(keypoints_array, axis=0)

    predictions_exercise = model_exercise.predict(keypoints_array)
    predicted_exercise_label = np.argmax(predictions_exercise, axis=1)[0]
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

    predictions_form = model_form.predict(keypoints_array)
    predicted_form_label = np.argmax(predictions_form, axis=1)[0]
    if predicted_form_label == 0:
        exercise_form = "technika ćwiczenia jest niepoprawna"
    else:
        exercise_form = "technika ćwiczenia jest poprawna"
    exercise = exercise_name +" a " + exercise_form
    return exercise, exercise_image
interface = gr.Interface(
    fn=predict,
    inputs=gr.Video(),
    outputs=[
        gr.Text(label="Opis"),
        gr.Image(label="Zaangażowane mięśnie",type="filepath")],
    title="Predykcja nazwy oraz techniki ćwiczenia  z wideo",
    description="Załaduj wideo, aby  rozpoznąć ćwiczenie oraz ocenić technikę"
)

interface.launch()
