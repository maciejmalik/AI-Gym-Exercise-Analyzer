import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

videos_df = pd.read_csv("file_names.csv", sep=";")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
def extract_keypoints(results):
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(132)
for exercise in videos_df["exercise"].unique():
    output_folder = os.path.join(f"{exercise} keypoints")
    filtered_df = videos_df[videos_df["exercise"] == exercise]
    for label in filtered_df["label"].unique():
        filtered_videos = filtered_df[filtered_df["label"] == label]

        for idx, row in filtered_videos.iterrows():
            os.makedirs(os.path.join(output_folder, str(label), str(idx)), exist_ok=True)
max_frames = 0
for idx, video in videos_df.iterrows():
    input_folder = f"{video['exercise']}"
    video_path = os.path.join(input_folder, video["filename"])
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = max(num_frames, max_frames)
    cap.release()
for exercise in videos_df["exercise"].unique():
    input_folder = f"{exercise}/"
    output_folder = os.path.join(f"{exercise} keypoints")
    filtered_df = videos_df[videos_df["exercise"] == exercise]
    for idx, video in filtered_df.iterrows():
        video_path = os.path.join(input_folder, video["filename"])
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in range(max_frames):
            if num_frames > frame_num:
                ret, frame = cap.read()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = pose.process(frame_rgb)
                keypoints = extract_keypoints(results)
            else:
                keypoints = np.zeros(132)
            npy_path = os.path.join(output_folder, str(video["label"]), str(idx), str(frame_num))
            np.save(npy_path, keypoints)
        cap.release()
pose.close()
cv2.destroyAllWindows()