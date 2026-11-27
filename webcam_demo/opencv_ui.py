"""Webcam window for computers where the optional Tk DLLs cannot load."""

import os
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import WebcamApp, extract_coordinates


class TextValue:
    """Small value holder for the prediction controller's text fields."""

    def __init__(self, value=''):
        self.value = value

    def set(self, value):
        self.value = value


class OpenCVWebcamApp(WebcamApp):
    """Reuse the Tk app's live prediction and trial logic with an OpenCV view."""

    WINDOW = 'Thai Sign Language - Space: reset/start | Esc: quit'

    def __init__(self, predictor, camera_index, window_seconds, word_test=None, category_sign_ids=None):
        self.predictor = predictor
        self.camera_index = camera_index
        self.window_seconds = window_seconds
        self.word_test = word_test
        self.category_sign_ids = category_sign_ids
        self.samples = deque()
        self.last_hand_time = None
        self.last_prediction_time = 0.0
        self.last_frame_time = None
        self.closed = False
        self.camera = None
        self.holistic = None
        for field in ('status', 'word', 'confidence', 'sign_id', 'timing',
                      'test_result', 'test_summary'):
            setattr(self, field, TextValue())
        self.confidence_bar = {'value': 0}
        self.clear_prediction()
        if word_test is not None:
            self.test_summary.set(word_test.summary())
        font_path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'tahoma.ttf'
        try:
            self.font = ImageFont.truetype(str(font_path), 20)
        except OSError:
            self.font = ImageFont.load_default(size=20)

    def render(self, frame, result):
        drawing = mp.solutions.drawing_utils
        for hand, color in ((result.left_hand_landmarks, (60, 230, 90)),
                            (result.right_hand_landmarks, (255, 180, 40))):
            drawing.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                drawing.DrawingSpec(color=color, thickness=2),
            )
        preview = Image.fromarray(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
        canvas = Image.new('RGB', (preview.width + 430, max(preview.height, 620)), '#19232b')
        canvas.paste(preview, (0, 0))
        draw = ImageDraw.Draw(canvas)
        x, y = preview.width + 16, 16
        lines = ['Thai Sign Language', 'Space: reset / start trial', 'Esc: quit']
        if self.word_test is not None:
            lines.append(f"Target: {self.word_test.target['thai']}")
        lines.extend(getattr(self, field).value for field in (
            'word', 'confidence', 'sign_id', 'status', 'timing', 'test_result', 'test_summary'))
        for text in lines:
            # Wrap by rendered width, including Thai text without spaces.
            for paragraph in str(text).split('\n'):
                line = ''
                for character in paragraph:
                    if line and draw.textlength(line + character, font=self.font) > 395:
                        draw.text((x, y), line, font=self.font, fill='white')
                        y += 28
                        line = ''
                    line += character
                draw.text((x, y), line, font=self.font, fill='white')
                y += 28
            y += 8
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def run(self):
        try:
            # Validate the window backend before requesting camera access.
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera.release()
                self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f'Cannot open camera {self.camera_index}; try --camera 1')
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.holistic = mp.solutions.holistic.Holistic(
                model_complexity=1, smooth_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
            while not self.closed:
                ok, frame = self.camera.read()
                if not ok:
                    raise RuntimeError('Cannot read webcam frame')
                captured_at = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = self.holistic.process(rgb)
                coordinates = extract_coordinates(result)
                if self.word_test is None:
                    self.update_live(captured_at, coordinates, result)
                else:
                    self.update_word_test(captured_at, coordinates)
                if self.last_frame_time is None:
                    print(f'Webcam active: {frame.shape[1]}x{frame.shape[0]}, '
                          'MediaPipe processed, OpenCV preview displayed', flush=True)
                self.last_frame_time = captured_at
                cv2.imshow(self.WINDOW, self.render(frame, result))
                key = cv2.waitKey(10) & 0xff
                if key == 27 or cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == 32:
                    self.reset()
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.camera is not None:
            self.camera.release()
        if self.holistic is not None:
            self.holistic.close()
        cv2.destroyAllWindows()
