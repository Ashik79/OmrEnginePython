import cv2
import numpy as np
import json
from omr_engine import OmrEngine

engine = OmrEngine(target_width=800, target_height=1000)

width = 800
height = 1000

# intercept extract_roi_data to modify the `warped` image BEFORE it is converted to gray and thresholded
original_extract = engine.extract_roi_data

def my_extract(self, warped, active_q=60):
    print("Monkey patched extract running!")
    # DRAW THE BUBBLES ON warped DIRECTLY BEFORE the rest of the function!
    # SET box: No Set (not filled)
    
    # ROLL number: 123456
    roll_digits = [1, 2, 3, 4, 5, 6]
    for col, digit in enumerate(roll_digits):
        bx = int(width * (0.072 + col * 0.07))
        by = int(height * (0.165 + digit * 0.0165))
        cv2.circle(warped, (bx, by), 10, (0, 0, 0), -1)

    # Questions 1 to 20 answers
    answers = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    for i in range(20):
        col_idx = i // 20
        row_idx = i % 20
        col_base_x = 0.02 if col_idx == 0 else (0.35 if col_idx == 1 else 0.68)
        base_y = int(height * (0.365 + row_idx * 0.0305 + (row_idx // 5) * 0.0085))
        opt_idx = answers[i]
        bx = int(width * (col_base_x + 0.08 + opt_idx * 0.055))
        by = base_y
        cv2.circle(warped, (bx, by), 10, (0, 0, 0), -1)

    # Now call the original extract
    return original_extract(warped, active_q)

import types
engine.extract_roi_data = types.MethodType(my_extract, engine)

# Create a blank image 800x1000
img = np.ones((1000, 800, 3), dtype=np.uint8) * 255
# Draw 4 registration marks (40x40 squares)
cv2.rectangle(img, (10, 10), (50, 50), (0,0,0), -1)
cv2.rectangle(img, (750, 10), (790, 50), (0,0,0), -1)
cv2.rectangle(img, (10, 950), (50, 990), (0,0,0), -1)
cv2.rectangle(img, (750, 950), (790, 990), (0,0,0), -1)
cv2.imwrite("blank.jpg", img)

print("Running engine on blank.jpg...")
try:
    data = engine.run("blank.jpg", active_q=20, output_json="fake_results.json", debug_output="fake_debug.jpg")
    print("SUCCESS")
    print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)
