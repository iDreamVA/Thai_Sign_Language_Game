"""Run a local webcam with the exported TSL-51 LSTM model."""

import argparse
import json
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageTk

from tsl_inference import SignPredictor
from tsl_preprocess import COORDINATE_COLUMNS, INPUT_FEATURES
from word_test import WordTest, resolve_target


APP_DIR = Path(__file__).resolve().parent
POSE_INDICES = (11, 12, 13, 14, 15, 16)
# TSL-51 dataset card lists the face subset in this order.
FACE_INDICES = (105, 70, 300, 334, 61, 291)


def extract_coordinates(result):
    """Left hand, right hand, six pose points, six face points -> [54, 3]."""
    coordinates = np.full((54, 3), np.nan, dtype=np.float32)
    for start, landmarks, indices in (
        (0, result.left_hand_landmarks, range(21)),
        (21, result.right_hand_landmarks, range(21)),
        (42, result.pose_landmarks, POSE_INDICES),
        (48, result.face_landmarks, FACE_INDICES),
    ):
        if landmarks is not None:
            for offset, index in enumerate(indices):
                point = landmarks.landmark[index]
                coordinates[start + offset] = (point.x, point.y, point.z)
    return coordinates


def load_predictor(model_dir):
    predictor = SignPredictor(model_dir)
    metadata = predictor.metadata
    if metadata['coordinate_columns'] != COORDINATE_COLUMNS or metadata['input_features'] != INPUT_FEATURES:
        raise ValueError('Model landmark layout differs from this webcam extractor')
    if metadata['sequence_frames'] < 2 or not metadata['classes']:
        raise ValueError('Invalid model metadata')
    if [item['index'] for item in metadata['classes']] != list(range(len(metadata['classes']))):
        raise ValueError('Class indexes are not in model-output order')
    with torch.inference_mode():
        logits = predictor.model(torch.zeros(1, metadata['sequence_frames'], INPUT_FEATURES))
    if logits.shape != (1, len(metadata['classes'])) or not torch.isfinite(logits).all():
        raise ValueError('Model output does not match metadata')
    return predictor


class WebcamApp:
    def __init__(self, root, predictor, camera_index, window_seconds, word_test=None):
        self.root = root
        self.predictor = predictor
        self.window_seconds = window_seconds
        self.word_test = word_test
        self.samples = deque()
        self.last_hand_time = None
        self.last_prediction_time = 0.0
        self.last_frame_time = None
        self.closed = False
        self.holistic = None
        self.camera = None
        self.status = tk.StringVar(value='กำลังเปิดกล้อง...')
        self.word = tk.StringVar(value='รอเริ่มทำท่า')
        self.confidence = tk.StringVar(value='ความมั่นใจ — %')
        self.sign_id = tk.StringVar(value='')
        self.timing = tk.StringVar(value='')
        self.test_result = tk.StringVar(value='')
        self.test_summary = tk.StringVar(value='')

        root.title('ทดลองภาษามือไทย · MediaPipe + LSTM')
        root.option_add('*Font', ('Tahoma', 11))
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Escape>', lambda event: self.close())
        root.bind('<space>', lambda event: self.reset())
        container = ttk.Frame(root, padding=16)
        container.pack(fill='both', expand=True)
        self.preview = ttk.Label(container)
        self.preview.grid(row=0, column=0, sticky='nsew')
        panel = ttk.Frame(container, padding=(20, 0, 0, 0))
        panel.grid(row=0, column=1, sticky='n')
        if word_test is not None:
            ttk.Label(panel, text=f"คำเป้าหมาย {word_test.target['thai']}", font=('Tahoma', 18),
                      wraplength=320).pack(anchor='w', pady=8)
            ttk.Label(panel, text=word_test.target['sign_id'], wraplength=320).pack(anchor='w')
        ttk.Label(panel, text='ท่าที่โมเดลทำนาย', font=('Tahoma', 16)).pack(anchor='w', pady=(12, 16))
        ttk.Label(panel, textvariable=self.word, font=('Tahoma', 28), wraplength=320).pack(anchor='w')
        ttk.Label(panel, textvariable=self.confidence, font=('Tahoma', 20)).pack(anchor='w', pady=16)
        self.confidence_bar = ttk.Progressbar(panel, maximum=100, length=300)
        self.confidence_bar.pack(anchor='w')
        ttk.Label(panel, textvariable=self.sign_id, wraplength=320).pack(anchor='w', pady=(12, 24))
        ttk.Label(panel, textvariable=self.status, wraplength=320).pack(anchor='w')
        ttk.Label(panel, textvariable=self.timing).pack(anchor='w', pady=12)
        if word_test is not None:
            ttk.Label(panel, textvariable=self.test_result, font=('Tahoma', 14), wraplength=320).pack(anchor='w')
            ttk.Label(panel, textvariable=self.test_summary, wraplength=320).pack(anchor='w', pady=8)
            self.test_summary.set(word_test.summary())
        ttk.Label(panel, text=f'ทำครั้งละหนึ่งท่าในช่วงประมาณ {window_seconds:g} วินาที\nให้เห็นใบหน้า ไหล่ และมือทั้งสองข้าง',
                  wraplength=320).pack(anchor='w', pady=12)
        ttk.Label(panel, text='เปอร์เซ็นต์คือความมั่นใจของคำทำนาย', wraplength=320).pack(anchor='w', pady=8)
        button_text = 'เริ่มรอบทดสอบ (Space)' if word_test is not None else 'เริ่มจับท่าใหม่ (Space)'
        ttk.Button(panel, text=button_text, command=self.reset).pack(fill='x', pady=(20, 8))
        ttk.Button(panel, text='ปิดกล้อง (Esc)', command=self.close).pack(fill='x')

        try:
            self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera.release()
                self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f'เปิดกล้องหมายเลข {camera_index} ไม่ได้ ลองปิดแอปที่ใช้กล้อง หรือเปลี่ยน --camera 1')
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.holistic = mp.solutions.holistic.Holistic(
                model_complexity=1, smooth_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
        except Exception:
            self.close()
            raise
        self.root.after(0, self.update_frame)

    def reset(self):
        if self.word_test is not None:
            if not self.word_test.active and not self.word_test.complete:
                self.clear_prediction()
                self.test_result.set('')
                self.word_test.start(time.perf_counter())
            return
        self.samples.clear()
        self.last_hand_time = None
        self.clear_prediction()
        self.status.set('ยกมือแล้วเริ่มทำท่าใหม่')

    def clear_prediction(self):
        self.word.set('รอเริ่มทำท่า')
        self.confidence.set('ความมั่นใจ — %')
        self.confidence_bar['value'] = 0
        self.sign_id.set('')

    def update_frame(self):
        if self.closed:
            return
        try:
            ok, frame = self.camera.read()
            if not ok:
                raise RuntimeError('อ่านภาพจากกล้องไม่ได้ โปรดตรวจสายกล้องหรือแอปที่ใช้งานกล้อง')
            captured_at = time.perf_counter()
            first_frame = self.last_frame_time is None
            # Infer from the original image. Mirror only the displayed preview.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self.holistic.process(rgb)
            coordinates = extract_coordinates(result)
            if self.word_test is not None:
                self.update_word_test(captured_at, coordinates)
            else:
                self.update_live(captured_at, coordinates, result)
            self.last_frame_time = captured_at

            drawing = mp.solutions.drawing_utils
            for hand, color in ((result.left_hand_landmarks, (60, 230, 90)),
                                (result.right_hand_landmarks, (255, 180, 40))):
                drawing.draw_landmarks(
                    frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                    drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    drawing.DrawingSpec(color=color, thickness=2),
                )
            if result.pose_landmarks is not None:
                height, width = frame.shape[:2]
                for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16)):
                    left, right = result.pose_landmarks.landmark[a], result.pose_landmarks.landmark[b]
                    if min(left.visibility, right.visibility) >= 0.5:
                        cv2.line(frame, (int(left.x * width), int(left.y * height)),
                                 (int(right.x * width), int(right.y * height)), (180, 180, 180), 2)
            image = Image.fromarray(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.preview_image)
            if first_frame:
                print(f'Webcam active: {frame.shape[1]}x{frame.shape[0]}, MediaPipe processed, preview displayed', flush=True)
        except Exception as exc:
            self.close()
            messagebox.showerror('เกิดข้อผิดพลาด', str(exc), parent=self.root)
            return
        self.root.after(10, self.update_frame)

    def update_live(self, captured_at, coordinates, result):
        has_hands = result.left_hand_landmarks is not None or result.right_hand_landmarks is not None
        if self.last_frame_time is not None and captured_at - self.last_frame_time > 0.6:
            self.samples.clear()
            self.clear_prediction()
        if has_hands:
            self.last_hand_time = captured_at
        if self.last_hand_time is None or captured_at - self.last_hand_time > 0.6:
            self.samples.clear()
            self.clear_prediction()
            self.status.set('ยังไม่พบมือ ยกมือให้อยู่ในภาพ')
        else:
            self.samples.append((captured_at, coordinates))
            while len(self.samples) > 2 and self.samples[1][0] < captured_at - self.window_seconds:
                self.samples.popleft()
            elapsed = captured_at - self.samples[0][0]
            self.status.set(f'กำลังเก็บท่า {min(elapsed, self.window_seconds):.1f} / {self.window_seconds:g} วินาที')
            if elapsed >= self.window_seconds and captured_at - self.last_prediction_time >= 0.3:
                self.predict()
                self.last_prediction_time = captured_at


    def update_word_test(self, captured_at, coordinates):
        test = self.word_test
        state, elapsed = test.capture(captured_at, coordinates)
        if state == 'idle':
            if test.complete:
                self.status.set(f'ครบ {test.trials} รอบแล้ว ปิดหน้าต่างเพื่อเปลี่ยนคำในเทอร์มินัล')
            else:
                self.status.set('กด Space เพื่อเริ่มรอบ มีเวลาเตรียมตัว 1.5 วินาที')
        elif state == 'countdown':
            self.status.set(f'เตรียมทำท่า {test.target["thai"]} ใน {elapsed:.1f} วินาที')
        elif state == 'recording':
            self.status.set(f'กำลังจับท่า {elapsed:.1f}/{test.seconds:g} วินาที')
        elif state == 'invalid':
            self.status.set('จับภาพไม่ครบหรือไม่พบมือ รอบนี้ยังไม่นับ กด Space เพื่อลองใหม่')
            self.test_result.set('ภาพไม่ครบหรือไม่พบมือ รอบนี้ยังไม่นับ')
        elif state == 'ready':
            prediction = test.predict(self.predictor)
            row = test.save_result(prediction)
            self.display_prediction(prediction)
            verdict = 'ตรงคำเป้าหมาย' if row['matched'] else 'ทายไม่ตรงคำเป้าหมาย'
            self.test_result.set(f"{verdict}\nคะแนนคำเป้าหมาย {row['target_probability'] * 100:.1f}%")
            self.test_summary.set(test.summary())
            self.status.set('บันทึกผลรอบนี้แล้ว กด Space เพื่อทดสอบรอบถัดไป')
            print(f"รอบ {row['trial']}/{test.trials} | เป้าหมาย {row['target_thai']} | "
                  f"ทำนาย {row['predicted_thai']} | มั่นใจ {row['top_confidence'] * 100:.1f}% | "
                  f"{verdict} | {test.summary()}", flush=True)
            print(f'ผลทดสอบ {test.path}', flush=True)

    def predict(self):
        stamps = np.array([sample[0] for sample in self.samples])
        coordinates = np.stack([sample[1] for sample in self.samples])
        started = time.perf_counter()
        prediction = self.predictor.predict_landmarks(coordinates, (stamps - stamps[0]) * 1000)
        self.display_prediction(prediction)
        self.status.set('ทำนายจากช่วงท่าล่าสุด ทำท่าถัดไปได้เลย')
        sample_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        self.timing.set(f'จับภาพ {sample_fps:.1f} FPS · LSTM {(time.perf_counter() - started) * 1000:.0f} ms')

    def display_prediction(self, prediction):
        confidence = prediction['raw_confidence'] * 100
        if prediction['rejected_low_confidence']:
            self.word.set('ยังไม่มั่นใจ')
            self.sign_id.set(f"คำที่ใกล้ที่สุด {prediction['raw_sign_id']}")
        else:
            self.word.set(prediction['thai'])
            self.sign_id.set(prediction['sign_id'])
        self.confidence.set(f'ความมั่นใจ {confidence:.1f}%')
        self.confidence_bar['value'] = confidence

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.camera is not None:
            self.camera.release()
        if self.holistic is not None:
            self.holistic.close()
        self.root.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=APP_DIR / 'model')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--window-seconds', type=float, default=2.0)
    parser.add_argument('--self-check', action='store_true', help='Check model and landmark pipeline without opening the camera')
    parser.add_argument('--list-words', action='store_true', help='List class names and exact sign IDs')
    parser.add_argument('--target', help='Thai target word or exact sign_id for one-word testing')
    parser.add_argument('--trials', type=int, default=10, help='Number of separately recorded target trials')
    parser.add_argument('--results-dir', type=Path, default=APP_DIR / 'sessions')
    args = parser.parse_args()
    if not 0.5 <= args.window_seconds <= 10:
        parser.error('--window-seconds must be between 0.5 and 10')
    if args.trials < 1:
        parser.error('--trials must be at least 1')
    metadata = json.loads((args.model_dir / 'metadata.json').read_text(encoding='utf-8'))
    if args.list_words:
        for item in metadata['classes']:
            print(f"{item['thai']}\t{item['sign_id']}")
        return
    target = None
    if args.target:
        try:
            target = resolve_target(metadata['classes'], args.target)
        except ValueError as exc:
            parser.error(str(exc))
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    predictor = load_predictor(args.model_dir)
    if args.self_check:
        with mp.solutions.holistic.Holistic(model_complexity=1) as detector:
            blank = detector.process(np.zeros((480, 640, 3), dtype=np.uint8))
            assert np.isnan(extract_coordinates(blank)).all()
        print(f"OK: MediaPipe, TorchScript, {len(predictor.metadata['classes'])} classes, "
              f"{predictor.metadata['sequence_frames']} frames, {INPUT_FEATURES} features")
        return
    root = tk.Tk()
    word_test = None
    if target is not None:
        word_test = WordTest(target, args.trials, args.window_seconds, args.results_dir,
                             args.camera, metadata['confidence_threshold'])
        print(f"ทดสอบคำ {target['thai']} ({target['sign_id']}) จำนวน {args.trials} รอบ กด Space เพื่อเริ่ม", flush=True)
    app = None
    try:
        app = WebcamApp(root, predictor, args.camera, args.window_seconds, word_test)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror('เริ่มโปรแกรมไม่ได้', str(exc), parent=root)
        raise
    finally:
        if app is not None:
            app.close()
        root.destroy()


if __name__ == '__main__':
    main()
