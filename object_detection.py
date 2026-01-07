import cv2
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import defaultdict


# ===================== CONFIG =====================
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.5
CAMERA_INDEX = 1  

FRAME_WIDTH = 640
FRAME_HEIGHT = 360


print("Loading YOLOv8 Nano model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully\n")


def choose_video_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()

    file_path = askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()
    return file_path


# ===================== VIDEO DETECTION =====================
def run_video_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Unable to open video file")

    detections = []
    frame_count = 0
    start_time = time.time()

    print("Video detection started. Press 'Q' to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        frame_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results = model(frame, stream=True)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

                detections.append(
                    [frame_count, timestamp, label, conf, x1, y1, x2, y2]
                )

        cv2.imshow("YOLOv8 Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    return detections, frame_count, fps


# ===================== WEBCAM DETECTION =====================
def run_webcam_detection():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    detections = []
    frame_count = 0
    start_time = time.time()

    print("Webcam detection started. Press 'Q' to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        frame_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results = model(frame, stream=True)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

                detections.append(
                    [frame_count, timestamp, label, conf, x1, y1, x2, y2]
                )

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        cv2.imshow("YOLOv8 Webcam Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    return detections, frame_count, fps


# ===================== TXT REPORT =====================
def generate_txt_report(mode, model_path, frames, fps, df, csv_file):
    report_file = "performance_report.txt"

    class_counts = df["Class"].value_counts()
    class_confidence = df.groupby("Class")["Confidence"].mean()

    with open(report_file, "w") as f:
        f.write("REAL-TIME OBJECT DETECTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mode               : {mode}\n")
        f.write(f"Model Used         : {model_path}\n")
        f.write(f"Total Frames       : {frames}\n")
        f.write(f"Total Detections   : {len(df)}\n")
        f.write(f"Average Confidence : {df['Confidence'].mean():.2f}\n")
        f.write(f"FPS Performance    : {fps:.2f}\n")
        f.write(f"CSV Log File       : {csv_file}\n")
        f.write(f"Generated On       : {datetime.now()}\n\n")

        f.write("PER-CLASS STATISTICS\n")
        f.write("-" * 50 + "\n")
        for cls in class_counts.index:
            f.write(
                f"{cls:<15} | Count: {class_counts[cls]:<6} "
                f"| Avg Conf: {class_confidence[cls]:.2f}\n"
            )

    return report_file


# ===================== MAIN =====================
def main():
    print("Select input type:")
    print("1. Video file")
    print("2. Webcam")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("[INFO] Please select a video file...")
        video_path = choose_video_file()
        if not video_path:
            print("❌ No file selected. Exiting.")
            return
        detections, frames, fps = run_video_detection(video_path)
        mode = "VIDEO"

    elif choice == "2":
        detections, frames, fps = run_webcam_detection()
        mode = "WEBCAM"

    else:
        print("❌ Invalid choice")
        return

    df = pd.DataFrame(
        detections,
        columns=["Frame_ID", "Timestamp", "Class",
                 "Confidence", "x1", "y1", "x2", "y2"]
    )

    csv_file = "detection_logs.csv"
    df.to_csv(csv_file, index=False)

    report_file = generate_txt_report(
        mode, MODEL_PATH, frames, fps, df, csv_file
    )

    print("\n================ FINAL REPORT ================")
    print(f"Mode            : {mode}")
    print(f"Model Used      : {MODEL_PATH}")
    print(f"Total Frames    : {frames}")
    print(f"Total Detections: {len(df)}")
    print(f"FPS             : {fps:.2f}")
    print(f"Overall Avg Conf: {df['Confidence'].mean():.2f}")
    print(f"CSV Log Saved   : {csv_file}")
    print(f"TXT Report Saved: {report_file}")
    print("==============================================\n")


if __name__ == "__main__":
    main()
