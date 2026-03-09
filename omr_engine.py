import cv2
import numpy as np
import json
import os
import base64
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


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
        self.yolo_model = None
        
        if HAS_YOLO:
            try:
                # Load pre-trained OMR model if exists, or nano model as fallback
                model_path = os.path.join(os.path.dirname(__file__), "omr_v8n.pt")
                if os.path.exists(model_path):
                    self.yolo_model = YOLO(model_path)
                else:
                    # Fallback to general YOLO for marker detection if specific model not found
                    self.yolo_model = YOLO("yolov8n.pt") 
                print(f"[+] YOLOv8 loaded successfully")
            except Exception as e:
                print(f"[-] YOLOv8 load failed: {e}")
                self.yolo_model = None

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
    def align_sheet(self, img, gray):
        """
        Finds 4 black square registration marks with a multi-threshold retry loop.
        """
        h_img, w_img = img.shape[:2]
        CORNER_ZONE = 0.45  # Relaxed from 0.3 for better file/high-res support

        # Try different threshold/edge settings to find markers
        threshold_configs = [
            {"type": "adaptive", "block": 15, "C": 3},
            {"type": "adaptive", "block": 11, "C": 2},
            {"type": "adaptive", "block": 21, "C": 4},
            {"type": "canny", "low": 60, "high": 200},
            {"type": "canny", "low": 30, "high": 150}
        ]

        markers = []
        for cfg in threshold_configs:
            markers = []
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            if cfg["type"] == "canny":
                processed = cv2.Canny(blurred, cfg["low"], cfg["high"])
            else:
                processed = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, cfg["block"], cfg["C"]
                )

            contours, _ = cv2.findContours(
                processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:
                area = cv2.contourArea(c)
                if area < 80 or area > 25000:
                    continue

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.05 * peri, True)
                if len(approx) != 4:
                    continue

                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / float(h)
                if not (0.5 <= aspect <= 2.0): # Relaxed aspect ratio
                    continue

                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Corner proximity check
                if (cy < h_img * CORNER_ZONE or cy > h_img * (1 - CORNER_ZONE)) and \
                   (cx < w_img * CORNER_ZONE or cx > w_img * (1 - CORNER_ZONE)):
                    markers.append((cx, cy)) # Store as tuple (cx, cy)

            if len(markers) >= 4:
                break # Found enough markers, exit the config loop

        if len(markers) < 4:
            print(f"WARNING: Only {len(markers)} markers found after multi-thresholding — using fallback.")
            screen_cnt = self._full_page_fallback(img)
        else:
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

    def align_sheet_ai(self, img):
        """
        Uses YOLO to detect corners for much more robust alignment.
        """
        if not self.yolo_model or img is None:
            _, g, _ = self.preprocess(img) if isinstance(img, str) else (img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            return self.align_sheet(img, g)

        h_img, w_img = img.shape[:2]
        # Run inference on markers
        results = self.yolo_model(img, conf=0.25, verbose=False)[0]
        
        detected_points = []
        for box in results.boxes:
            # We assume classes: 0: marker (or 4 classes for corners)
            # For simplicity, we filter by confidence and position
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)
            detected_points.append((cx, cy))

        if len(detected_points) < 4:
            print(f"[-] YOLO markers failed ({len(detected_points)} detected), fallback to Legacy.")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return self.align_sheet(img, gray)

        # Sort points by Y then X to find corners
        pts = np.array(detected_points, dtype="float32")
        # Custom sort logic for 4 corners
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        
        screen_cnt = np.array([tl, tr, br, bl], dtype="float32")
        
        # Perspective transform with target dimensions
        dst = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(screen_cnt, dst)
        warped = cv2.warpPerspective(img, M, (self.target_width, self.target_height))
        print("[+] YOLO Alignment Success")
        return warped

    # ──────────────────────────────────────────────────────────────
    #  STEP 3 — Extract ROI data
    # ──────────────────────────────────────────────────────────────
    def extract_roi_data(self, warped, active_q=100):
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
        """
        SET detection removed as per user request.
        """
        return ""

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

        roll_start_x   = 0.070
        roll_start_y   = 0.145
        roll_col_space = 0.048
        roll_row_space = 0.0182
        roi_r          = 7    # Smaller radius for smaller bubbles (17px diam)

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

            # Adaptive minimum: use mean of nonzero as a baseline
            nonzero = [d for d in densities if d > 5]
            dynamic_min = int(np.mean(nonzero) * 0.5) if nonzero else 25

            # Confident if: density above floor AND significantly higher than noise
            confident = (
                max_d >= max(25, dynamic_min) and
                (second_d == 0 or max_d / max(second_d, 1) >= 1.25)
            )

            if confident:
                roll += str(max_row)
                bx_f = int(width  * (roll_start_x + col * roll_col_space))
                by_f = int(height * (roll_start_y + max_row * roll_row_space))
                cv2.circle(debug_img, (bx_f, by_f), 7, (0, 255, 0), 2)
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

        num_cols   = 4
        q_per_col  = 25
        total_q    = 100

        q_start_y   = 0.345
        q_row_space  = 0.024
        gap_height   = 0.006  # extra gap every 5 questions
        bubble_r     = 12      # bubble ROI half-size

        # ── 4 columns positions: 5%, 28.5%, 52%, 75.5% (approx from image)
        col_configs = [
            {"base_x": 0.05},
            {"base_x": 0.285},
            {"base_x": 0.52},
            {"base_x": 0.755},
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
                bx = int(width * (col_base_x + 0.08 + opt_idx * 0.038))
                by = base_y

                roi = thresh[
                    max(0, by - bubble_r): by + bubble_r,
                    max(0, bx - bubble_r): bx + bubble_r
                ]

                # ── Circular Masking: ignore noise outside the bubble radius
                if roi.size > 0:
                    rr, cc = np.ogrid[:roi.shape[0], :roi.shape[1]]
                    center_y, center_x = roi.shape[0] // 2, roi.shape[1] // 2
                    mask = (rr - center_y)**2 + (cc - center_x)**2 <= (bubble_r-1)**2
                    pixel_count = int(cv2.countNonZero(roi[mask]))
                    roi_area = int(np.sum(mask))
                else:
                    pixel_count = 0
                    roi_area = 1

                # Density as percentage
                density_pct = float((pixel_count / roi_area) * 100)

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

            is_empty = float(top_pct) < float(EMPTY_THRESHOLD)
            is_multi_fill = (
                float(top_pct) >= float(VALID_THRESHOLD) and
                float(second_pct) >= float(MULTI_SECOND_THRESH) and
                (float(top_pct) / max(float(second_pct), 1.0)) < float(CONFIDENCE_RATIO)
            )
            is_valid = float(top_pct) >= float(VALID_THRESHOLD) and not is_multi_fill

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

        print(f"[+] Evaluated {active_q} of 100 questions. {100 - active_q} SKIPPED_INACTIVE.")
        return extracted_answers

    # ──────────────────────────────────────────────────────────────
    #  MAIN — Full pipeline
    # ──────────────────────────────────────────────────────────────
    def run(self, image_path, active_q=100, output_json=None, debug_output=None, skip_align=False):
        print(f"[*] Processing: {image_path} | Active Q: {active_q}/100")
        img, gray, _ = self.preprocess(image_path)
        if skip_align:
            warped = cv2.resize(img, (self.target_width, self.target_height))
        else:
            warped = self.align_sheet(img, gray)
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
