import cv2
import numpy as np
import json
import csv
import os
import base64

class OmrEngine:
    """
    Universal OMR Engine — works with the fixed 60-question SOHAG PHYSICS sheet.
    
    The sheet ALWAYS has 60 questions printed. The `active_q` parameter tells
    the engine how many of those questions to actually evaluate. The remaining
    bubbles are completely ignored in scoring. This way one printed sheet works
    for exams of any size: 10, 15, 20, 25 … 60.
    """

    def __init__(self, target_width=800, target_height=1000):
        self.target_width = target_width
        self.target_height = target_height
        self.option_labels = ['A', 'B', 'C', 'D']

    def preprocess(self, image_path):
        """Loads and prepares the image for alignment with CLAHE."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE to fix shadows BEFORE finding registration marks
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        return img, gray, edged

    def align_sheet(self, img, edged):
        """Finds the 4 black square registration marks and performs perspective transform."""
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markers = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            area = cv2.contourArea(c)

            if len(approx) == 4 and 200 < area < 10000:
                x, y, w, h = cv2.boundingRect(approx)
                aspectRatio = w / float(h)
                if 0.7 <= aspectRatio <= 1.3:
                    markers.append(c)

        if len(markers) >= 4:
            points = []
            for m in markers:
                M = cv2.moments(m)
                if M["m00"] == 0: continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy))

            if len(points) < 4:
                # Fallback to page bounds if moments failed
                print("WARNING: Moment calculation failed for markers. Falling back.")
                h_img, w_img = img.shape[:2]
                screen_cnt = np.array([[w_img-10, 10], [10, 10], [10, h_img-10], [w_img-10, h_img-10]], dtype="float32")
            else:
                # Sort points: top-left, top-right, bottom-right, bottom-left
                points = sorted(points, key=lambda p: p[1])
                top_points = sorted(points[:2], key=lambda p: p[0])
                bottom_points = sorted(points[-2:], key=lambda p: p[0])
                screen_cnt = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype="float32")
        else:
            print("WARNING: Could not find exactly 4 registration marks! Falling back to page boundaries.")
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            x, y, w, h = cv2.boundingRect(contours[0])
            screen_cnt = np.array([[x+w, y], [x, y], [x, y+h], [x+w, y+h]], dtype="float32")

        rect = np.zeros((4, 2), dtype="float32")
        s = screen_cnt.sum(axis=1)
        rect[0] = screen_cnt[np.argmin(s)]
        rect[2] = screen_cnt[np.argmax(s)]
        diff = np.diff(screen_cnt, axis=1)
        rect[1] = screen_cnt[np.argmin(diff)]
        rect[3] = screen_cnt[np.argmax(diff)]

        dst = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (self.target_width, self.target_height))
        return warped

    def extract_roi_data(self, warped, active_q=60):
        """
        Processes specific ROIs and detects bubbles.
        
        active_q: int — how many of the 60 questions to evaluate.
                  Questions beyond active_q are returned as SKIPPED (ignored).
        """
        debug_img = warped.copy()
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Apply intense CLAHE on warped image for bubble clarity
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_warped = clahe.apply(gray_warped)

        blurred = cv2.GaussianBlur(gray_warped, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

        detected_set = self._process_set_section(thresh, debug_img)
        if detected_set:
            print(f"[+] SET Code Detected: {detected_set}")
            cv2.putText(debug_img, f"SET: {detected_set}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Extraction logic
        results = {
            "set": detected_set,
            "roll": self._process_roll_section(thresh, debug_img),
            "questions": self._process_questions_section(thresh, debug_img, active_q=active_q)
        }

        # --- STEP 3: ROLL CROP PREVIEW ---
        # Capture a tight crop of the Roll number area for UI verification
        roll_box_y1 = int(self.target_height * 0.12)
        roll_box_y2 = int(self.target_height * 0.35)
        roll_box_x1 = int(self.target_width * 0.04)
        roll_box_x2 = int(self.target_width * 0.50)
        roll_crop = warped[roll_box_y1:roll_box_y2, roll_box_x1:roll_box_x2]
        _, roll_buffer = cv2.imencode(".jpg", roll_crop)
        results["roll_crop_base64"] = "data:image/jpeg;base64," + base64.b64encode(roll_buffer).decode()

        return results, debug_img

    def _process_set_section(self, thresh, debug_img):
        """
        Set detection has been removed from the OMR sheet.
        Always returns None.
        """
        return None

    def _process_roll_section(self, thresh, debug_img):
        """Detects roll number with dynamic relative percentages."""
        roll = ""
        width = self.target_width
        height = self.target_height

        roll_start_x = 0.072
        roll_start_y = 0.165
        roll_col_space = 0.07
        roll_row_space = 0.0165

        for col in range(6):
            densities = []
            best_digit = None
            max_d = 0
            second_max_d = 0

            for row in range(10):
                bx = int(width * (roll_start_x + col * roll_col_space))
                by = int(height * (roll_start_y + row * roll_row_space))

                roi = thresh[by-10:by+10, bx-10:bx+10]
                d = cv2.countNonZero(roi)
                densities.append(d)
                cv2.rectangle(debug_img, (bx-10, by-10), (bx+10, by+10), (150, 150, 150), 1)

                if d > max_d:
                    second_max_d = max_d
                    max_d = d
                    best_digit = row
                elif d > second_max_d:
                    second_max_d = d

            if max_d < 50 or (max_d > 50 and second_max_d > 40 and (max_d / max(second_max_d, 1)) < 1.3):
                roll += "?"
            else:
                roll += str(best_digit) if best_digit is not None else "?"

            if best_digit is not None:
                final_bx = int(width * (roll_start_x + col * roll_col_space))
                final_by = int(height * (roll_start_y + best_digit * roll_row_space))
                cv2.circle(debug_img, (final_bx, final_by), 8, (0, 255, 0), 2)

        return roll

    def _process_questions_section(self, thresh, debug_img, active_q=60):
        """
        Processes the FIXED 60-question bubble grid.
        
        UNIVERSAL SHEET LOGIC:
          - Sheet always has 60 questions (3 columns × 20)
          - Only questions 1..active_q are evaluated for scoring
          - Questions active_q+1..60 are returned as SKIPPED (not scored)
          - This lets you reuse one printed sheet for 10, 15, 20 ... 60 Q exams
        """
        extracted_answers = []
        width = self.target_width
        height = self.target_height

        # Fixed 60-MCQ grid geometry (3 columns × 20 rows)
        num_cols = 3
        q_per_col = 20

        q_start_y = 0.365
        q_row_space = 0.0305
        gap_height = 0.0085   # extra gap every 5 questions

        total_q = 60  # ALWAYS process all 60 positions for debug image
        extracted_answers = []

        for i in range(total_q):
            col_idx = i // q_per_col
            row_idx = i % q_per_col

            col_base_x = 0.02 if col_idx == 0 else (0.35 if col_idx == 1 else 0.68)
            base_y = int(height * (q_start_y + row_idx * q_row_space + (row_idx // 5) * gap_height))

            q_number = i + 1
            is_active = q_number <= active_q

            # Draw debug boxes — grey for inactive, normal for active
            debug_color = (180, 180, 180) if is_active else (220, 220, 220)

            options_data = []
            for opt_idx in range(4):
                bx = int(width * (col_base_x + 0.08 + opt_idx * 0.055))
                by = base_y

                roi = thresh[by-12:by+12, bx-12:bx+12]
                pixel_count = cv2.countNonZero(roi)
                options_data.append({"opt": self.option_labels[opt_idx], "pixels": pixel_count, "bx": bx, "by": by})
                cv2.rectangle(debug_img, (bx-12, by-12), (bx+12, by+12), debug_color, 1)

            # ── SKIP evaluation for questions beyond active_q ──────────
            if not is_active:
                extracted_answers.append({
                    "qNum": q_number,
                    "detected": None,
                    "isError": False,
                    "errorType": "SKIPPED_INACTIVE"
                })
                continue

            # ── Normal evaluation for active questions ─────────────────
            sorted_opts = sorted(options_data, key=lambda x: int(x["pixels"]), reverse=True)
            highest = int(sorted_opts[0]["pixels"])
            second_highest = int(sorted_opts[1]["pixels"])

            is_multi_fill = highest > 100 and (second_highest > 60 and (highest / max(second_highest, 1)) < 1.4)
            is_empty = highest < 80

            detected = sorted_opts[0]["opt"] if (highest >= 80 and not is_multi_fill) else None

            if is_multi_fill:
                status = "MULTIPLE_FILL"
                cv2.putText(debug_img, "!", (int(width * col_base_x), base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            elif is_empty:
                status = "EMPTY"
            else:
                status = "VALID"
                cv2.circle(debug_img, (sorted_opts[0]["bx"], sorted_opts[0]["by"]), 10, (255, 0, 0), 2)

            extracted_answers.append({
                "qNum": q_number,
                "detected": detected,
                "isError": is_multi_fill or is_empty,
                "errorType": "MULTIPLE_FILL" if is_multi_fill else ("EMPTY" if is_empty else None)
            })

        print(f"[+] Evaluated {active_q} questions. Remaining {60 - active_q} marked SKIPPED_INACTIVE.")
        return extracted_answers

    def run(self, image_path, active_q=60, output_json="results.json", debug_output="debug_scan.jpg"):
        """
        Main processing pipeline.
        
        active_q: Number of questions to actually evaluate (5–60).
                  Default 60 = evaluate all. Set to exam size for partial eval.
        """
        print(f"[*] Processing: {image_path} | Active Questions: {active_q}/60")
        img, gray, edged = self.preprocess(image_path)
        warped = self.align_sheet(img, edged)
        data, debug_img = self.extract_roi_data(warped, active_q=active_q)

        with open(output_json, 'w') as f:
            json.dump(data, f, indent=4)

        cv2.imwrite(debug_output, debug_img)
        print(f"[+] Processing Complete! Roll: {data['roll']} | SET: {data['set']}")
        print(f"[+] Debug image saved to {debug_output}")
        return data


if __name__ == "__main__":
    engine = OmrEngine()
    # Example: evaluate only 25 questions from the 60-bubble sheet
    # engine.run("sample_omr.jpg", active_q=25)
    print("OMR Engine Initialized. Universal 60-Q Sheet Ready.")
    print("Usage: engine.run('sheet.jpg', active_q=25)  # evaluate only 25 of 60")
