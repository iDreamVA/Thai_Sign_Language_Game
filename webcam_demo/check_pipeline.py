"""Verify the live-to-training feature contract without accessing a webcam."""

from types import SimpleNamespace
import csv
import tempfile
from pathlib import Path

import numpy as np

from app import APP_DIR, FACE_INDICES, POSE_INDICES, extract_coordinates, group_classes, load_predictor
from tsl_preprocess import landmarks_to_tensor
from word_test import WordTest, resolve_target


def landmark_list(count):
    return SimpleNamespace(landmark=[
        SimpleNamespace(x=i / 500, y=i / 600, z=-i / 700) for i in range(count)
    ])


def main():
    result = SimpleNamespace(left_hand_landmarks=landmark_list(21), right_hand_landmarks=None,
                             pose_landmarks=landmark_list(33), face_landmarks=landmark_list(468))
    coordinates = extract_coordinates(result)
    assert coordinates.shape == (54, 3)
    assert np.isnan(coordinates[21:42]).all()
    np.testing.assert_allclose(coordinates[42:48, 0], np.array(POSE_INDICES) / 500)
    np.testing.assert_allclose(coordinates[48:54, 0], np.array(FACE_INDICES) / 500)
    sequence = np.stack([coordinates] * 12)
    tensor = landmarks_to_tensor(sequence, 48, np.arange(12) * 100)
    assert tensor.shape == (48, 216) and np.isfinite(tensor).all()
    assert not tensor[:, 63:126].any() and not tensor[:, 183:204].any()
    predictor = load_predictor(APP_DIR / 'model')
    prediction = predictor.predict_landmarks(sequence, np.arange(12) * 100)
    assert prediction['sign_id'] in {item['sign_id'] for item in predictor.metadata['classes']}
    assert 0 <= prediction['raw_confidence'] <= 1
    assert len(prediction['probabilities']) == len(predictor.metadata['classes'])
    assert np.isclose(sum(prediction['probabilities']), 1.0)
    groups = group_classes(predictor.metadata)
    assert {name: len({item['thai'] for item in items}) for name, items in groups.items()} == {
        'greetings': 4, 'family': 8, 'questions': 3, 'daily': 11, 'feelings': 6, 'other': 18,
    }
    assert {item['sign_id'] for item in groups['family'] if item['thai'] == 'ฉัน'} == {'ฉัน_var_1', 'ฉัน_var_2'}
    assert 'กรุงเทพ' in {item['thai'] for item in groups['other']}
    assert sorted(item['index'] for items in groups.values() for item in items) == [
        item['index'] for item in predictor.metadata['classes'] if item['index'] != predictor.metadata['null_index']
    ]
    target = resolve_target(predictor.metadata['classes'], 'ขอบคุณ')
    assert target['sign_id'] == 'ขอบคุณ_เปิดมือสองข้าง'
    try:
        resolve_target(predictor.metadata['classes'], 'ฉัน')
    except ValueError:
        pass
    else:
        raise AssertionError('Ambiguous Thai word was accepted')
    with tempfile.TemporaryDirectory() as directory:
        test = WordTest(target, 2, 0.5, directory, 0, 0)
        test.start(0)
        assert test.capture(0.5, coordinates)[0] == 'countdown'
        assert test.capture(1.5, coordinates)[0] == 'recording'
        assert test.capture(1.8, coordinates)[0] == 'recording'
        assert test.capture(2.0, coordinates)[0] == 'ready'
        assert test.capture(2.1, coordinates)[0] == 'idle'
        test.predict(predictor)
        scores = [0.0] * len(predictor.metadata['classes'])
        scores[target['index']] = 1.0
        correct = {'sign_id': target['sign_id'], 'thai': target['thai'],
                   'raw_sign_id': target['sign_id'], 'raw_confidence': 1.0,
                   'rejected_low_confidence': False, 'probabilities': scores}
        assert test.save_result(correct)['matched'] == 1
        test.start(3)
        assert test.capture(4.5, coordinates)[0] == 'recording'
        assert test.capture(4.8, coordinates)[0] == 'recording'
        assert test.capture(5.0, coordinates)[0] == 'ready'
        incorrect = {**correct, 'sign_id': 'null_act', 'thai': 'ไม่พบรูปท่า'}
        assert test.save_result(incorrect)['matched'] == 0
        assert test.complete and '50.0%' in test.summary()
        test.start(6)
        assert not test.active
        with test.path.open(encoding='utf-8-sig', newline='') as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == 2 and [row['matched'] for row in saved] == ['1', '0']
        invalid = WordTest(target, 1, 0.5, directory, 0, 0)
        invalid.start(0)
        missing = np.full((54, 3), np.nan)
        invalid.capture(1.5, missing)
        invalid.capture(1.8, missing)
        assert invalid.capture(2.0, missing)[0] == 'invalid'
        assert not invalid.results and not invalid.path.exists()
        invalid.start(3)
        invalid.capture(4.5, coordinates)
        assert invalid.capture(5.3, coordinates)[0] == 'invalid'
    print('PASS: exact landmark order, missing-hand masks, preprocessing, exported model prediction')
    print('PASS: category coverage, both variants of ฉัน, and exclusion of the null class')
    print('PASS: target resolution, countdown, separate trials, correct/wrong outcomes, CSV and invalid captures')


if __name__ == '__main__':
    main()
