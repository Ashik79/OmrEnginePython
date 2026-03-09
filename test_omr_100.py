import cv2
import numpy as np
import json
import base64
from omr_engine import OmrEngine

def create_fake_omr_100():
    width = 800
    height = 1000
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    def fill_bubble(x, y, radius=10):
        cv2.circle(img, (int(x), int(y)), radius, (0, 0, 0), -1)

    # Add registration marks (4 black squares at corners)
    # We draw them so their centers are at (20,20), (width-20, 20), etc.
    # to match the 800x1000 coordinate space correctly after alignment.
    marker_size = 40
    # top-left
    cv2.rectangle(img, (0, 0), (marker_size, marker_size), (0, 0, 0), -1)
    # top-right
    cv2.rectangle(img, (width - marker_size, 0), (width, marker_size), (0, 0, 0), -1)
    # bottom-left
    cv2.rectangle(img, (0, height - marker_size), (40, height), (0, 0, 0), -1)
    # bottom-right
    cv2.rectangle(img, (width - marker_size, height - marker_size), (width, height), (0, 0, 0), -1)

    # ROLL number: 123456
    # Coordinates from engine: roll_start_x = 0.070, roll_start_y = 0.135, col_space = 0.048, row_space = 0.0182
    roll_start_x = 0.070
    roll_start_y = 0.148
    roll_col_space = 0.048
    roll_row_space = 0.0182
    roll_digits = [1, 2, 3, 4, 5, 6]
    for col, digit in enumerate(roll_digits):
        bx = width * (roll_start_x + col * roll_col_space)
        by = height * (roll_start_y + digit * roll_row_space)
        fill_bubble(bx, by, radius=7)

    # Questions 1 to 100 answers
    # Loop pattern: A, B, C, D repeats
    # Coordinates from engine: 
    # q_start_y = 0.33, q_row_space = 0.024, gap_height = 0.006
    # col_base_x = [0.05, 0.285, 0.52, 0.755]
    # bx = width * (col_base_x + 0.08 + opt_idx * 0.038)
    q_start_y = 0.342
    q_row_space = 0.0248
    gap_height = 0.003
    # Column configurations from engine
    col_configs = [
        {"base_x": 0.050, "y_offset": 0},
        {"base_x": 0.285, "y_offset": 0},
        {"base_x": 0.520, "y_offset": -1}, # Sync with engine
        {"base_x": 0.755, "y_offset": -1}, # Sync with engine
    ]
    
    expected_answers = []
    
    for i in range(100):
        col_idx = i // 25
        row_idx = i % 25
        opt_idx = i % 4
        
        cfg = col_configs[col_idx]
        col_base_x = cfg["base_x"]
        col_y_offset = cfg["y_offset"]

        base_y = height * (q_start_y + row_idx * q_row_space + (row_idx // 5) * gap_height) + col_y_offset
        
        bx = width * (col_base_x + 0.08 + opt_idx * 0.038)
        by = base_y
        fill_bubble(bx, by, radius=10)
        expected_answers.append(['A', 'B', 'C', 'D'][opt_idx])

    cv2.imwrite("fake_omr_100.jpg", img)
    return expected_answers

def test_engine():
    expected_answers = create_fake_omr_100()
    engine = OmrEngine(target_width=800, target_height=1000)
    
    print("Running engine on fake_omr_100.jpg (skip_align=True)...")
    data = engine.run("fake_omr_100.jpg", active_q=100, output_json="fake_results_100.json", debug_output="fake_debug_100.jpg", skip_align=True)
    
    # Verify Roll
    if data['roll'] == "123456":
        print("[PASS] Roll number detected correctly: 123456")
    else:
        print(f"[FAIL] Roll number detected: {data['roll']}, expected: 123456")

    # Verify Questions
    mismatches = []
    for i, q in enumerate(data['questions']):
        detected = q['detected']
        expected = expected_answers[i]
        if detected != expected:
            mismatches.append(f"Q{i+1}: expected {expected}, got {detected} (Error: {q['errorType']})")
    
    if not mismatches:
        print("[PASS] All 100 questions detected correctly!")
    else:
        print(f"[FAIL] {len(mismatches)} mismatches found:")
        for m in mismatches[:10]:
            print(f"  {m}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches)-10} more")

if __name__ == "__main__":
    test_engine()
