"""One completed signing interval is one target-word trial."""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np


def resolve_target(classes, target):
    exact = [item for item in classes if item['sign_id'] == target]
    matches = exact or [item for item in classes if item['thai'] == target]
    if len(matches) != 1:
        options = ', '.join(item['sign_id'] for item in matches)
        raise ValueError(f'คำนี้มีหลายรูปท่า ให้ระบุ sign_id: {options}' if matches
                         else f'ไม่พบคำ {target!r} ใช้ --list-words เพื่อดูคำที่โมเดลรองรับ')
    return matches[0]


class WordTest:
    def __init__(self, target, trials, seconds, output_dir, camera_index, threshold):
        self.target = target
        self.trials = trials
        self.seconds = seconds
        self.output_dir = Path(output_dir)
        self.camera_index = camera_index
        self.threshold = threshold
        self.path = self.output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_word_test.csv"
        self.results = []
        self.samples = []
        self.starts_at = None
        self.last_frame_at = None

    @property
    def active(self):
        return self.starts_at is not None

    @property
    def complete(self):
        return len(self.results) >= self.trials

    def start(self, now):
        if self.complete or self.active:
            return
        self.samples = []
        self.starts_at = now + 1.5
        self.last_frame_at = None

    def capture(self, now, coordinates):
        """Return countdown, recording, ready, or invalid, with elapsed seconds."""
        if not self.active:
            return 'idle', 0
        if now < self.starts_at:
            return 'countdown', self.starts_at - now
        if self.last_frame_at is not None and now - self.last_frame_at > 0.6:
            self.starts_at = None
            return 'invalid', 0
        self.last_frame_at = now
        self.samples.append((now, coordinates))
        elapsed = now - self.starts_at
        if elapsed < self.seconds:
            return 'recording', elapsed
        self.starts_at = None
        visible_hands = any(np.isfinite(points[:42]).all(axis=-1).any() for _, points in self.samples)
        visible_points = any(np.isfinite(points).all(axis=-1).any() for _, points in self.samples)
        expected_rest = self.target['sign_id'] == 'null_act'
        if len(self.samples) < 2 or not visible_points or (not visible_hands and not expected_rest):
            return 'invalid', elapsed
        return 'ready', elapsed

    def predict(self, predictor):
        timestamps = np.array([sample[0] for sample in self.samples])
        points = np.stack([sample[1] for sample in self.samples])
        return predictor.predict_landmarks(points, (timestamps - timestamps[0]) * 1000)

    def save_result(self, prediction):
        matched = (not prediction['rejected_low_confidence']
                   and prediction['sign_id'] == self.target['sign_id'])
        row = {
            'trial': len(self.results) + 1,
            'timestamp': datetime.now().astimezone().isoformat(),
            'target_sign_id': self.target['sign_id'], 'target_thai': self.target['thai'],
            'predicted_sign_id': prediction['sign_id'], 'predicted_thai': prediction['thai'],
            'raw_sign_id': prediction['raw_sign_id'],
            'top_confidence': prediction['raw_confidence'],
            'target_probability': prediction['probabilities'][self.target['index']],
            'matched': int(matched),
            'rejected_low_confidence': int(prediction['rejected_low_confidence']),
            'camera': self.camera_index, 'window_seconds': self.seconds,
            'threshold': self.threshold, 'captured_frames': len(self.samples),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        with self.path.open('a', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self.results.append(row)
        self.samples = []
        return row

    def summary(self):
        hits = sum(row['matched'] for row in self.results)
        total = len(self.results)
        rate = f'{100 * hits / total:.1f}%' if total else '—'
        return f'ตรงคำเป้าหมาย {hits}/{total} รอบ ({rate})'
