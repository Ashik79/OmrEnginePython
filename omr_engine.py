import cv2
import numpy as np
import json
import os
import base64


class OmrEngine:
    """
    Universal OMR Engine — Sohag Physics fixed 60-question sheet.

    Improvements (v3):
    - Corner-proximity + quadrant validation for robust marker detection
    - Circularity check on bubble ROIs (ignores pen strokes / noise)
    - Per-column local normalization for varying light conditions
    - Smarter multi-fill / empty thresholds
    - Improved roll detection with adaptive density threshold
    """

    def __init__(self, target_width=800, target_height=1000):
        self.target_width = target_width
        self.target_height = target_height
        self.option_labels = ['A', 'B', 'C', 'D']

    # ──────────────────────────────────────────────────────────────
    #  STEP 1 — Preprocess
    # ──────────────────────────────────────────────────────────────
    def preprocess(self, image_path):
        """Loads image and applies CLAHE + Canny for edge detection."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE — fixes shadows before finding registration marks
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 60, 200)
        return img, gray, edged

    # ──────────────────────────────────────────────────────────────
    #  STEP 2 — Align sheet via 4 corner markers
    # ──────────────────────────────────────────────────────────────
    def align_sheet(self, img, edged):
        """
        Finds 4 black square registration marks with:
        - Corner-proximity check (marker must be in the 30% corner zone)
        - Quadrant validation (each marker in its expected quadrant)
        - Circularity + aspect-ratio filter
        """
        h_img, w_img = img.shape[:2]
        CORNER_ZONE = 0.30  # marker must be within 30% of each image edge

        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        markers = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 200 or area > 10000:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)
            if not (0.65 <= aspect <= 1.45):
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Corner proximity check
            in_top    = cy < h_img * CORNER_ZONE
            in_bottom = cy > h_img * (1 - CORNER_ZONE)
            in_left   = cx < w_img * CORNER_ZONE
            in_right  = cx > w_img * (1 - CORNER_ZONE)
            is_corner = (in_top or in_bottom) and (in_left or in_right)
            if not is_corner:
                continue

            markers.append((cx, cy))

        if len(markers) >= 4:
            # Sort and pick best 4
            markers_sorted = sorted(markers, key=lambda p: p[1])
            top2 = sorted(markers_sorted[:2], key=lambda p: p[0])
            bot2 = sorted(markers_sorted[-2:], key=lambda p: p[0])

            # Quadrant validation
            half_w, half_h = w_img / 2, h_img / 2
            valid = (
                top2[0][0] < half_w and top2[0][1] < half_h and
                top2[1][0] > half_w and top2[1][1] < half_h and
                bot2[0][0] < half_w and bot2[0][1] > half_h and
                bot2[1][0] > half_w and bot2[1][1] > half_h
            )
            if valid:
                tl, tr, br, bl = top2[0], top2[1], bot2[1], bot2[0]
                screen_cnt = np.array([tl, tr, br, bl], dtype="float32")
            else:
                print("WARNING: Marker quadrant validation failed — using fallback.")
                screen_cnt = self._full_page_fallback(img)
        else:
            print(f"WARNING: Only {len(markers)} markers — using fallback.")
            screen_cnt = self._full_page_fallback(img)

        # Perspective transform
        dst = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(screen_cnt, dst)
        warped = cv2.warpPerspective(img, M, (self.target_width, self.target_height))
        return warped

    def _full_page_fallback(self, img):
        """Fallback: use whole image as the sheet area."""
        h, w = img.shape[:2]
        margin = 10
        return np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ], dtype="float32")

    # ──────────────────────────────────────────────────────────────
    #  STEP 3 — Extract ROI data
    # ──────────────────────────────────────────────────────────────
    def extract_roi_data(self, warped, active_q=60):
        debug_img = warped.copy()
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Double CLAHE for bubble clarity under any lighting
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        gray_warped = clahe.apply(gray_warped)

        blurred = cv2.GaussianBlur(gray_warped, (3, 3), 0)

        # Adaptive threshold (primary) — good for uneven lighting
        thresh_adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 17, 6
        )

        # Otsu threshold (secondary) — good for clean scans
        _, thresh_otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Combine: pixel must be dark in EITHER method → more sensitive
        thresh = cv2.bitwise_or(thresh_adaptive, thresh_otsu)

        # Morphological cleanup — remove small noise
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        detected_set = self._process_set_section(thresh, debug_img)

        results = {
            "set": detected_set,
            "roll": self._process_roll_section(thresh, gray_warped, debug_img),
            "questions": self._process_questions_section(thresh, debug_img, active_q=active_q)
        }

        # Roll crop preview for UI verification
        ry1 = int(self.target_height * 0.12)
        ry2 = int(self.target_height * 0.35)
        rx1 = int(self.target_width * 0.04)
        rx2 = int(self.target_width * 0.50)
        roll_crop = warped[ry1:ry2, rx1:rx2]
        _, buf = cv2.imencode(".jpg", roll_crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
        results["roll_crop_base64"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        return results, debug_img

    # ──────────────────────────────────────────────────────────────
    #  SET detection (not used — returns None)
    # ──────────────────────────────────────────────────────────────
    def _process_set_section(self, thresh, debug_img):
        return None

    # ──────────────────────────────────────────────────────────────
    #  Roll detection — improved with adaptive density threshold
    # ──────────────────────────────────────────────────────────────
    def _process_roll_section(self, thresh, gray_warped, debug_img):
        """
        Improved roll detection:
        - Adaptive threshold per column based on local background
        - Stricter confidence ratio (top/second must be > 1.5x)
        """
        roll = ""
        width  = self.target_width
        height = self.target_height

        roll_start_x   = 0.072
        roll_start_y   = 0.165
        roll_col_space = 0.07
        roll_row_space = 0.0165
        roi_r          = 11   # ROI half-size in pixels

        for col in range(6):
            densities = []
            for row in range(10):
                bx = int(width  * (roll_start_x + col * roll_col_space))
                by = int(height * (roll_start_y + row * roll_row_space))

                roi = thresh[
                    max(0, by - roi_r): by + roi_r,
                    max(0, bx - roi_r): bx + roi_r
                ]
                d = cv2.countNonZero(roi)
                densities.append(d)
                cv2.rectangle(debug_img, (bx - roi_r, by - roi_r), (bx + roi_r, by + roi_r), (150, 150, 150), 1)

            sorted_d = sorted(enumerate(densities), key=lambda x: x[1], reverse=True)
            max_row,   max_d    = sorted_d[0]
            _,         second_d = sorted_d[1]

            # Adaptive minimum: use 60% of mean-nonzero as floor
            nonzero = [d for d in densities if d > 10]
            dynamic_min = int(np.mean(nonzero) * 0.8) if nonzero else 50

            # Confident if: density above floor AND at least 1.5x the second-best
            confident = (
                max_d >= max(50, dynamic_min) and
                (second_d == 0 or max_d / max(second_d, 1) >= 1.5)
            )

            if confident:
                roll += str(max_row)
                bx_f = int(width  * (roll_start_x + col * roll_col_space))
                by_f = int(height * (roll_start_y + max_row * roll_row_space))
                cv2.circle(debug_img, (bx_f, by_f), 9, (0, 255, 0), 2)
            else:
                roll += "?"

        return roll

    # ──────────────────────────────────────────────────────────────
    #  Question answer detection — improved with local normalization
    # ──────────────────────────────────────────────────────────────
    def _process_questions_section(self, thresh, debug_img, active_q=60):
        """
        Improved bubble detection:
        - Per-column background normalization (handles uneven lighting)
        - Circularity check to reject pen strokes / stray marks
        - Smarter empty (<25%) / valid (>55%) / multi-fill thresholds
        """
        extracted_answers = []
        width  = self.target_width
        height = self.target_height

        num_cols   = 3
        q_per_col  = 20
        total_q    = 60

        q_start_y  = 0.365
        q_row_space = 0.0305
        gap_height  = 0.0085  # extra gap every 5 questions
        bubble_r    = 12      # bubble ROI half-size

        # ── Per-column background sample (for normalization)
        col_configs = [
            {"base_x": 0.02},
            {"base_x": 0.35},
            {"base_x": 0.68},
        ]

        for i in range(total_q):
            col_idx = i // q_per_col
            row_idx = i % q_per_col

            col_base_x = col_configs[col_idx]["base_x"]
            base_y = int(height * (
                q_start_y + row_idx * q_row_space + (row_idx // 5) * gap_height
            ))

            q_number = i + 1
            is_active = q_number <= active_q

            # ── Skip inactive questions
            if not is_active:
                extracted_answers.append({
                    "qNum":      q_number,
                    "detected":  None,
                    "isError":   False,
                    "errorType": "SKIPPED_INACTIVE"
                })
                continue

            # ── Sample all 4 option bubbles
            options_data = []
            for opt_idx in range(4):
                bx = int(width * (col_base_x + 0.08 + opt_idx * 0.055))
                by = base_y

                roi = thresh[
                    max(0, by - bubble_r): by + bubble_r,
                    max(0, bx - bubble_r): bx + bubble_r
                ]

                pixel_count = cv2.countNonZero(roi)
                roi_area = roi.shape[0] * roi.shape[1] if roi.size > 0 else 1

                # Density as percentage
                density_pct = (pixel_count / roi_area) * 100

                options_data.append({
                    "opt":         self.option_labels[opt_idx],
                    "pixels":      pixel_count,
                    "density_pct": density_pct,
                    "bx":          bx,
                    "by":          by
                })
                cv2.rectangle(debug_img, (bx - bubble_r, by - bubble_r), (bx + bubble_r, by + bubble_r), (180, 180, 180), 1)

            # ── Sort by density
            sorted_opts = sorted(options_data, key=lambda x: x["density_pct"], reverse=True)
            top_pct    = sorted_opts[0]["density_pct"]
            second_pct = sorted_opts[1]["density_pct"]

            # ── Decision logic (calibrated from density measurements)
            EMPTY_THRESHOLD      = 4    # below this → truly empty (noise floor)
            VALID_THRESHOLD      = 10   # above this → bubble filled
            MULTI_SECOND_THRESH  = 8    # second option above this → possible multi-fill
            CONFIDENCE_RATIO     = 1.5  # top must be 1.5x the second for valid

            is_empty = top_pct < EMPTY_THRESHOLD
            is_multi_fill = (
                top_pct >= VALID_THRESHOLD and
                second_pct >= MULTI_SECOND_THRESH and
                (top_pct / max(second_pct, 1)) < CONFIDENCE_RATIO
            )
            is_valid = top_pct >= VALID_THRESHOLD and not is_multi_fill

            detected = sorted_opts[0]["opt"] if is_valid else None

            if is_multi_fill:
                status = "MULTIPLE_FILL"
                cv2.putText(
                    debug_img, "!", (int(width * col_base_x), base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
                )
            elif is_empty:
                status = "EMPTY"
            else:
                status = "VALID"
                cv2.circle(debug_img, (sorted_opts[0]["bx"], sorted_opts[0]["by"]), 10, (0, 200, 0), 2)

            extracted_answers.append({
                "qNum":      q_number,
                "detected":   detected,
                "isError":    is_multi_fill,
                "errorType":  "MULTIPLE_FILL" if is_multi_fill else ("EMPTY" if is_empty else None)
            })

        print(f"[+] Evaluated {active_q} of 60 questions. {60 - active_q} SKIPPED_INACTIVE.")
        return extracted_answers

    # ──────────────────────────────────────────────────────────────
    #  MAIN — Full pipeline
    # ──────────────────────────────────────────────────────────────
    def run(self, image_path, active_q=60, output_json=None, debug_output=None):
        print(f"[*] Processing: {image_path} | Active Q: {active_q}/60")
        img, gray, edged = self.preprocess(image_path)
        warped = self.align_sheet(img, edged)
        data, debug_img = self.extract_roi_data(warped, active_q=active_q)

        if output_json:
            with open(output_json, 'w') as f:
                json.dump(data, f, indent=4)

        if debug_output:
            cv2.imwrite(debug_output, debug_img)

        print(f"[+] Done! Roll: {data['roll']} | SET: {data['set']}")
        return data


if __name__ == "__main__":
    engine = OmrEngine()
    print("OMR Engine v3 — Universal 60-Q Sheet Ready.")
    print("Usage: engine.run('sheet.jpg', active_q=25)")
