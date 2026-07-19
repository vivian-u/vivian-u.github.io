#!/usr/bin/env python3
"""
Crop portraits to a uniform square centered on detected faces.

Usage:
  python3 crop_faces.py --src images/team --dst images/team_cropped --size 400

This script uses OpenCV's Haar cascade for face detection. If OpenCV is not
installed, install it with `pip install opencv-python`.
"""
import argparse
import os
import cv2
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_detector():
    # Use OpenCV's built-in haarcascade for frontal face
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if hasattr(cv2, 'CascadeClassifier') and os.path.exists(cascade_path):
        return cv2.CascadeClassifier(cascade_path)
    # fallback to a local copy next to this script
    local_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    if hasattr(cv2, 'CascadeClassifier') and os.path.exists(local_path):
        return cv2.CascadeClassifier(local_path)
    # Could not load a cascade detector (OpenCV build may omit it). Return None
    # and let the caller fall back to center cropping.
    return None


def detect_largest_face(detector, gray):
    if detector is None:
        return None
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # choose the largest face by area
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]
    return (x, y, w, h)


def crop_center_on_face(img, face, out_size):
    h, w = img.shape[:2]
    if face is None:
        # fallback: center crop
        cx, cy = w // 2, h // 2
    else:
        x, y, fw, fh = face
        cx = int(x + fw / 2)
        cy = int(y + fh / 2)

    half = out_size // 2
    left = cx - half
    top = cy - half
    right = cx + half
    bottom = cy + half

    # Adjust if crop goes out of bounds
    if left < 0:
        right += -left
        left = 0
    if top < 0:
        bottom += -top
        top = 0
    if right > w:
        left -= (right - w)
        right = w
    if bottom > h:
        top -= (bottom - h)
        bottom = h

    # final clamp
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    crop = img[top:bottom, left:right]

    # If crop size != out_size (image too small), pad evenly
    ch, cw = crop.shape[:2]
    if ch == out_size and cw == out_size:
        return crop

    pad_top = max(0, (out_size - ch) // 2)
    pad_bottom = max(0, out_size - ch - pad_top)
    pad_left = max(0, (out_size - cw) // 2)
    pad_right = max(0, out_size - cw - pad_left)

    color = [255, 255, 255]
    crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=color)
    return crop


def process_folder(src_dir, dst_dir, size):
    ensure_dir(dst_dir)
    detector = load_detector()

    for fname in sorted(os.listdir(src_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        img = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping unreadable image: {src_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detect_largest_face(detector, gray)
        cropped = crop_center_on_face(img, face, size)

        # encode and write reliably for paths with spaces
        success, buf = cv2.imencode('.jpg', cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if success:
            buf.tofile(dst_path)
            print(f"Saved: {dst_path}")
        else:
            print(f"Failed to encode: {dst_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Source folder with portrait images')
    parser.add_argument('--dst', required=True, help='Destination folder for cropped images')
    parser.add_argument('--size', type=int, default=400, help='Output square size in pixels')
    args = parser.parse_args()

    process_folder(args.src, args.dst, args.size)


if __name__ == '__main__':
    main()
