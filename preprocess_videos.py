import os
import cv2
import numpy as np
import argparse

def preprocess_videos(input_dir, output_dir, frame_size=(128, 128), sequence_len=8):
    """
    Preprocess raw videos into fixed-size sequences for training/testing.

    Args:
        input_dir (str): Directory containing raw videos.
        output_dir (str): Directory to save preprocessed sequences.
        frame_size (tuple): Target size (width, height) for each frame.
        sequence_len (int): Number of frames per sequence.
    """
    os.makedirs(output_dir, exist_ok=True)
    for video_name in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_name)
        if not video_path.endswith(('.mp4', '.avi', '.mov')):
            continue  # Skip non-video files
        cap = cv2.VideoCapture(video_path)
        frames, count = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, frame_size))
            if len(frames) == sequence_len:
                output_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_{count}.npy")
                np.save(output_path, np.array(frames))
                frames, count = [], count + 1
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw videos for REVMark.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the raw video directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save preprocessed video sequences.")
    parser.add_argument("--frame_size", type=int, nargs=2, default=(128, 128), help="Target size (width, height) for frames.")
    parser.add_argument("--sequence_len", type=int, default=8, help="Number of frames per video sequence.")

    args = parser.parse_args()
    preprocess_videos(args.input_dir, args.output_dir, tuple(args.frame_size), args.sequence_len)
