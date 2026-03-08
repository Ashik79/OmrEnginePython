import cv2
import numpy as np
import json
from omr_engine import OmrEngine

width = 800
height = 1000
img = np.ones((height, width, 3), dtype=np.uint8) * 255

def fill_bubble(x, y):
    cv2.circle(img, (int(x), int(y)), 10, (0, 0, 0), -1)

# Add registration marks (4 black squares at corners)
# top-left
cv2.rectangle(img, (10, 10), (35, 35), (0, 0, 0), -1)
# top-right
cv2.rectangle(img, (width - 35, 10), (width - 10, 35), (0, 0, 0), -1)
# bottom-left
cv2.rectangle(img, (10, height - 35), (35, height - 10), (0, 0, 0), -1)
# bottom-right
cv2.rectangle(img, (width - 35, height - 35), (width - 10, height - 10), (0, 0, 0), -1)

# SET box: B (row 1)
bx = 800 * 0.56
by = 1000 * (0.165 + 1 * 0.035)
fill_bubble(bx, by)

# ROLL number: 123456
roll_digits = [1, 2, 3, 4, 5, 6]
for col, digit in enumerate(roll_digits):
    bx = 800 * (0.072 + col * 0.07)
    by = 1000 * (0.165 + digit * 0.0165)
    fill_bubble(bx, by)

# Questions 1 to 20 answers
# 0=A, 1=B, 2=C, 3=D
answers = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
for i in range(20):
    col_idx = i // 20
    row_idx = i % 20
    col_base_x = 0.02 if col_idx == 0 else (0.35 if col_idx == 1 else 0.68)
    base_y = 1000 * (0.365 + row_idx * 0.0305 + (row_idx // 5) * 0.0085)
    
    opt_idx = answers[i]
    bx = 800 * (col_base_x + 0.08 + opt_idx * 0.055)
    by = base_y
    fill_bubble(bx, by)

cv2.imwrite("fake_omr_20.jpg", img)

engine = OmrEngine(target_width=width, target_height=height)

print("Running engine on fake_omr_20.jpg with 20 questions active...")
try:
    data = engine.run("fake_omr_20.jpg", active_q=20, output_json="fake_results.json", debug_output="fake_debug.jpg")
    print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)
