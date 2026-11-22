"""Print and export word-level results from the existing held-out test matrix."""

import argparse
import csv
import json
from pathlib import Path


def review(bundle, output):
    metadata = json.loads((bundle / 'metadata.json').read_text(encoding='utf-8'))
    classes = metadata['classes']
    labels = [item['sign_id'] for item in classes]
    with (bundle / 'tsl51_user_confusion.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header, data = next(reader), list(reader)
    if header[1:] != labels or [row[0] for row in data] != labels:
        raise ValueError('Confusion matrix order differs from metadata')
    matrix = [[int(value) for value in row[1:]] for row in data]
    if any(len(row) != len(labels) or any(value < 0 for value in row) for row in matrix):
        raise ValueError('Invalid confusion matrix')
    rows = []
    for i, item in enumerate(classes):
        correct = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * correct / (support + predicted) if support + predicted else 0.0
        errors = sorted((j for j in range(len(classes)) if j != i and matrix[i][j]),
                        key=lambda j: matrix[i][j], reverse=True)[:3]
        rows.append({
            'thai': item['thai'], 'sign_id': item['sign_id'], 'correct': correct,
            'test_clips': support, 'precision': precision, 'recall': recall, 'f1': f1,
            'frequent_confusions': ' | '.join(
                f"{classes[j]['sign_id']} ({matrix[i][j]})" for j in errors),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['sign_id']} | ถูก {row['correct']}/{row['test_clips']} | "
              f"P {row['precision'] * 100:.1f}% R {row['recall'] * 100:.1f}% F1 {row['f1'] * 100:.1f}%")
    total = sum(row['test_clips'] for row in rows)
    accuracy = sum(row['correct'] for row in rows) / total
    supported = [row for row in rows if row['test_clips']]
    print(f"\nAccuracy {accuracy * 100:.2f}% | Macro F1 {sum(row['f1'] for row in supported) / len(supported) * 100:.2f}%")
    print(f"Test clips {total} | Classes with zero recall {sum(row['correct'] == 0 for row in supported)}")
    print(f'Report saved to {output}')
    return rows


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=root.parent / 'tsl51_lstm_bundle')
    parser.add_argument('--output', type=Path, default=root / 'reports/model_word_report.csv')
    args = parser.parse_args()
    review(args.bundle, args.output)
