from datetime import timedelta

import cv2
import numpy as np
import tensorflow as tf


def extract_frames(video_path, interval, divisor):
    """
    Extract frames from a video at a specified interval, resize, and return as an array.

    Args:
        video_path (str): Path to the input video file.
        interval (int): Interval in seconds to capture frames.
        divisor (int): Factor by which the frame dimensions will be reduced.

    Returns:
        tuple: (frames, timestamps), where frames is a numpy array of processed frames
               and timestamps is a list of corresponding timestamps.
    """
    print("Loading video file...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, fps * interval)  # Modified to prevent becoming 0

    frames = []
    timestamps = []

    # Calculate total frames to process
    total_frames_to_process = len(range(0, total_frames, frame_interval))
    print(f"Extracting frames at interval of {frame_interval} from total {total_frames} frames...")
    processed_frames = 0

    for frame_idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to the frame position
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        height, width, _ = frame.shape
        resized_frame = cv2.resize(frame, (width // divisor, height // divisor))
        resized_frame = resized_frame.astype("float32") / 255.0  # Normalize

        frames.append(resized_frame)

        # Calculate timestamp (hh:mm:ss format)
        timestamp = str(timedelta(seconds=frame_idx // fps))
        timestamps.append(timestamp)

        processed_frames += 1
        progress = (processed_frames / total_frames_to_process) * 100
        if processed_frames % 10 == 0:  # Print progress every 10 frames
            print(f"Processing frames: {processed_frames}/{total_frames_to_process} ({progress:.1f}% complete)")

    cap.release()
    print("Frame extraction complete!")
    return np.array(frames), timestamps


def classify_frames(frames, model_path):
    """
    Classify frames using a pre-trained model.

    Args:
        frames (numpy array): Array of processed frames.
        model_path (str): Path to the trained model file.

    Returns:
        list: List of classification results (0 or 1).
    """
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print(f"Starting classification of {len(frames)} frames...")
    predictions = model.predict(frames, verbose=0)  # Set verbose=0 to hide default progress bar
    print("Frame classification complete!")
    return np.argmax(predictions, axis=1)  # Convert softmax output to 0 or 1


def analyze_video(video_path, interval, divisor, model_path):
    """
    Process the video and classify extracted frames.

    Args:
        video_path (str): Path to the input video file.
        interval (int): Interval in seconds to capture frames.
        divisor (int): Factor for resizing frames.
        model_path (str): Path to the trained model file.

    Returns:
        list: List of tuples (timestamp, classification result).
    """
    print("Starting video analysis...")
    frames, timestamps = extract_frames(video_path, interval, divisor)
    classifications = classify_frames(frames, model_path)
    print("Video analysis complete!")
    return list(zip(timestamps, [int(x) for x in classifications]))


def find_continuous_intervals(results):
    intervals = []
    start_time = None
    consecutive_count = 0
    interval_step = None

    for i in range(len(results)):
        timestamp, label = results[i]

        if label == 1:
            if start_time is None:
                start_time = timestamp
                consecutive_count = 1
                if i < len(results) - 1:
                    next_time = results[i + 1][0]
                    t1 = list(map(int, timestamp.split(":")))
                    t2 = list(map(int, next_time.split(":")))
                    interval_step = (t2[0] - t1[0]) * 3600 + (t2[1] - t1[1]) * 60 + (t2[2] - t1[2])
            else:
                consecutive_count += 1
        else:
            if start_time is not None:
                # Calculate end time of continuous interval
                end_time = start_time.split(":")
                h, m, s = map(int, end_time)
                total_seconds = h * 3600 + m * 60 + s + (interval_step * consecutive_count)
                h, m, s = total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
                end_time = f"{h:02d}:{m:02d}:{s:02d}"

                intervals.append((start_time, end_time))
                start_time = None
                consecutive_count = 0

    # Handle last interval
    if start_time is not None:
        h, m, s = map(int, start_time.split(":"))
        total_seconds = h * 3600 + m * 60 + s + (interval_step * consecutive_count)
        h, m, s = total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
        end_time = f"{h:02d}:{m:02d}:{s:02d}"
        intervals.append((start_time, end_time))

    return intervals


# Example usage
if __name__ == "__main__":
    model_path = input("Enter the path to the model file: ").strip('"')
    video_path = input("Enter the path to the video file: ").strip('"')
    interval = int(input("Enter frame extraction interval (seconds): "))
    divisor = int(input("Enter divisor for resizing: "))

    results = analyze_video(video_path, interval, divisor, model_path)
    print(results)

    # Save results to file
    output_file = "video_analysis_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{video_path}\n")
        for timestamp, result in results:
            f.write(f"{timestamp}, {result}\n")
    print(f"Analysis results have been saved to {output_file}")

    # Find continuous intervals of 1s
    continuous_intervals = find_continuous_intervals(results)
    print("Continuous intervals of 1s:")
    print(continuous_intervals)

    # Save continuous intervals to file
    output_file = "continuous_intervals.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{video_path}\n")
        for interval in continuous_intervals:
            f.write(f"{interval}\n")
    print(f"Continuous intervals have been saved to {output_file}")
