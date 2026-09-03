import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2
import PIL.Image as Image
import os

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            return func

print("Loading models...")
anatomy_model = YOLO("models/anatomy_model.pt")
dental_model = YOLO("models/dental_model.pt")
print("Models loaded successfully!")

ANATOMY_PRIORITY_CLASSES = {"Maxillary sinus", "Mandibular Canal"}
CLASS_ALIASES = {
    "Inferior Alveolar Nerve Canal": "Mandibular Canal",
    "Maxillary Sinus": "Maxillary sinus"
}

# Color palette (BGR format for OpenCV)
COLORS_BGR = [
    (75, 25, 230),    # Red #e6194b
    (75, 180, 60),    # Green #3cb44b
    (25, 225, 255),   # Yellow #ffe119
    (216, 99, 67),    # Blue #4363d8
    (49, 130, 245),   # Orange #f58231
    (180, 30, 145),   # Purple #911eb4
    (240, 240, 70),   # Cyan #46f0f0
    (230, 50, 240),   # Magenta #f032e6
    (12, 246, 188),   # Lime #bcf60c
    (190, 190, 250),  # Pink #fabebe
    (128, 128, 0),    # Teal #008080
    (255, 190, 230),  # Lavender #e6beff
    (36, 99, 154),    # Brown #9a6324
    (200, 250, 255),  # Beige #fffac8
    (0, 0, 128),      # Maroon #800000
    (195, 255, 170)   # Mint #aaffc3
]

def normalize_name(name):
    return CLASS_ALIASES.get(name, name)

def extract_detections(result, names_dict, source, img_w, img_h):
    detections = []
    if result.masks is None:
        return detections

    for mask, box in zip(result.masks.xy, result.boxes):
        cls_id = int(box.cls[0])
        class_name = normalize_name(names_dict[cls_id])
        confidence = float(box.conf[0])

        polygon = np.array(mask).astype(int).tolist()
        poly_np = np.array(polygon)
        
        if len(poly_np) == 0:
            continue

        cx = float(np.mean(poly_np[:, 0]))
        cy = float(np.mean(poly_np[:, 1]))

        # Calculate closest side (0: Left, 1: Right, 2: Top, 3: Bottom)
        d_left = cx
        d_right = img_w - cx
        d_top = cy
        d_bottom = img_h - cy
        side = int(np.argmin([d_left, d_right, d_top, d_bottom]))

        detections.append({
            "class_name": class_name,
            "confidence": confidence,
            "polygon": polygon,
            "source": source,
            "cx": cx,
            "cy": cy,
            "side": side
        })

    return detections

def merge_results(anatomy_dets, dental_dets):
    merged = []
    merged.extend(anatomy_dets)
    for det in dental_dets:
        if det["class_name"] in ANATOMY_PRIORITY_CLASSES:
            continue
        merged.append(det)
    return merged

def compute_anti_collision_callouts(detections, left_margin, top_margin, img_w, img_h, canvas_h):
    BADGE_R = 15
    MIN_GAP = 36

    sides = {0: [], 1: [], 2: [], 3: []}
    for d in detections:
        s = d.get("side", 0)
        sides[s].append(d)

    # Left (0) and Right (1)
    for side in [0, 1]:
        group = sides[side]
        if not group:
            continue
        group.sort(key=lambda d: d["cy"])

        y_min = top_margin + BADGE_R
        y_max = top_margin + img_h - BADGE_R
        pos = [int(d["cy"] + top_margin) for d in group]

        for i in range(1, len(pos)):
            if pos[i] < pos[i-1] + MIN_GAP:
                pos[i] = pos[i-1] + MIN_GAP

        if pos[-1] > y_max:
            pos[-1] = y_max
            for i in range(len(pos) - 2, -1, -1):
                if pos[i] > pos[i+1] - MIN_GAP:
                    pos[i] = pos[i+1] - MIN_GAP

        if pos[0] < y_min:
            pos[0] = y_min
            for i in range(1, len(pos)):
                if pos[i] < pos[i-1] + MIN_GAP:
                    pos[i] = pos[i-1] + MIN_GAP

        badge_x = left_margin - 70 if side == 0 else left_margin + img_w + 70
        elbow_x = left_margin - 25 if side == 0 else left_margin + img_w + 25

        for idx, d in enumerate(group):
            d["tx"] = badge_x
            d["ty"] = pos[idx]
            d["elbowX"] = elbow_x
            d["elbowY"] = int(d["cy"] + top_margin)

    # Top (2) and Bottom (3)
    for side in [2, 3]:
        group = sides[side]
        if not group:
            continue
        group.sort(key=lambda d: d["cx"])

        x_min = left_margin + BADGE_R
        x_max = left_margin + img_w - BADGE_R
        pos = [int(d["cx"] + left_margin) for d in group]

        for i in range(1, len(pos)):
            if pos[i] < pos[i-1] + MIN_GAP:
                pos[i] = pos[i-1] + MIN_GAP

        if pos[-1] > x_max:
            pos[-1] = x_max
            for i in range(len(pos) - 2, -1, -1):
                if pos[i] > pos[i+1] - MIN_GAP:
                    pos[i] = pos[i+1] - MIN_GAP

        if pos[0] < x_min:
            pos[0] = x_min
            for i in range(1, len(pos)):
                if pos[i] < pos[i-1] + MIN_GAP:
                    pos[i] = pos[i-1] + MIN_GAP

        badge_y = top_margin - 50 if side == 2 else top_margin + img_h + 50
        elbow_y = top_margin - 20 if side == 2 else top_margin + img_h + 20

        for idx, d in enumerate(group):
            d["tx"] = pos[idx]
            d["ty"] = badge_y
            d["elbowX"] = int(d["cx"] + left_margin)
            d["elbowY"] = elbow_y

@spaces.GPU
def predict_dental_xray(image):
    if image is None:
        return None, "Please upload an image."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    temp_path = "/tmp/temp_input.jpg"
    image.save(temp_path)

    # Run inference
    anatomy_result = anatomy_model.predict(source=temp_path, imgsz=1024, conf=0.25, verbose=False)[0]
    dental_result = dental_model.predict(source=temp_path, imgsz=1024, conf=0.25, verbose=False)[0]

    img_bgr = cv2.imread(temp_path)
    img_h, img_w, _ = img_bgr.shape

    anatomy_dets = extract_detections(anatomy_result, anatomy_model.names, "Anatomical", img_w, img_h)
    dental_dets = extract_detections(dental_result, dental_model.names, "Dental", img_w, img_h)
    final_detections = merge_results(anatomy_dets, dental_dets)

    for idx, det in enumerate(final_detections, start=1):
        det["id"] = idx
        det["color_bgr"] = COLORS_BGR[(idx - 1) % len(COLORS_BGR)]

    # Dynamic Canvas Setup
    LEFT_MARGIN = 220
    TOP_MARGIN = 100
    EXTRA_BOTTOM = 160
    
    num_dets = len(final_detections)
    max_per_col = 16
    num_cols = max(1, int(np.ceil(num_dets / max_per_col)))
    legend_width = max(380, num_cols * 210 + 40)

    canvas_w = img_w + LEFT_MARGIN + legend_width
    canvas_h = max(img_h + TOP_MARGIN + EXTRA_BOTTOM, 800)

    # Create off-white background canvas
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 250

    # Place image in canvas center-left
    canvas[TOP_MARGIN:TOP_MARGIN+img_h, LEFT_MARGIN:LEFT_MARGIN+img_w] = img_bgr

    # Draw Polygons over the image area
    for det in final_detections:
        color = det["color_bgr"]
        pts = np.array(det["polygon"], dtype=np.int32)
        pts[:, 0] += LEFT_MARGIN
        pts[:, 1] += TOP_MARGIN

        # Draw contour border
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Draw semi-transparent fill
        poly_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)

        roi = canvas[poly_mask == 255]
        blended = (roi.astype(np.float32) * 0.7 + np.array(color, dtype=np.float32) * 0.3).astype(np.uint8)
        canvas[poly_mask == 255] = blended

    # Compute callouts anti-collision layout
    compute_anti_collision_callouts(final_detections, LEFT_MARGIN, TOP_MARGIN, img_w, img_h, canvas_h)

    # Draw Orthogonal Callout Lines & Number Badges
    for det in final_detections:
        color = det["color_bgr"]
        cx = int(det["cx"] + LEFT_MARGIN)
        cy = int(det["cy"] + TOP_MARGIN)
        tx = int(det["tx"])
        ty = int(det["ty"])
        elbow_x = int(det["elbowX"])
        elbow_y = int(det["elbowY"])

        # Orthogonal line path
        if det["side"] in [0, 1]:
            pts_line = np.array([[cx, cy], [elbow_x, cy], [elbow_x, ty], [tx, ty]], np.int32)
        else:
            pts_line = np.array([[cx, cy], [cx, elbow_y], [tx, elbow_y], [tx, ty]], np.int32)

        cv2.polylines(canvas, [pts_line], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Centroid dot
        cv2.circle(canvas, (cx, cy), 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Badge circle at (tx, ty)
        cv2.circle(canvas, (tx, ty), 15, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 15, color, 2, cv2.LINE_AA)

        id_str = str(det["id"])
        text_size = cv2.getTextSize(id_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0]
        text_x = tx - text_size[0] // 2
        text_y = ty + text_size[1] // 2
        cv2.putText(canvas, id_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    # Draw Legend on the right side of Canvas
    start_x = LEFT_MARGIN + img_w + 110
    start_y = TOP_MARGIN

    cv2.putText(canvas, "Legend", (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (15, 23, 42), 2, cv2.LINE_AA)
    start_y += 35

    col_width = 205
    row_height = 28

    summary_rows = []

    for idx, det in enumerate(final_detections):
        col = idx // max_per_col
        row = idx % max_per_col

        item_x = start_x + col * col_width
        item_y = start_y + row * row_height

        color = det["color_bgr"]

        # Legend item colored badge circle
        cv2.circle(canvas, (item_x + 10, item_y + 10), 10, color, -1, cv2.LINE_AA)

        # ID inside circle
        id_str = str(det["id"])
        ts = cv2.getTextSize(id_str, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        cv2.putText(canvas, id_str, (item_x + 10 - ts[0]//2, item_y + 10 + ts[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Class Label text
        label = det["class_name"]
        if len(label) > 20:
            label = label[:18] + ".."
        cv2.putText(canvas, label, (item_x + 28, item_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (51, 65, 85), 1, cv2.LINE_AA)

        conf_pct = f"{det['confidence']:.1%}"
        summary_rows.append(f"| **#{det['id']}** | **{det['class_name']}** | {det['source']} | `{conf_pct}` |")

    # Convert BGR back to RGB for Gradio output
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    summary_text = (
        "### 📋 Detections Summary Table\n\n"
        "| ID | Class Name | Category | Confidence |\n"
        "| :---: | :--- | :---: | :---: |\n" +
        "\n".join(summary_rows)
    ) if summary_rows else "No detections found."

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return canvas_rgb, summary_text

# Define Gradio Interface
demo = gr.Interface(
    fn=predict_dental_xray,
    inputs=gr.Image(type="pil", label="Upload Dental X-Ray Image"),
    outputs=[
        gr.Image(type="numpy", label="Dental Segmentation Canvas Output (Callouts & Legend)"),
        gr.Markdown(label="Detections Summary Table")
    ],
    title="🦷 Dental AI Segmentation Viewer",
    description="Upload a panoramic dental X-ray to perform dual AI instance segmentation. Generates a full canvas visualization with anti-collision callout badges (1, 2, 3...) on margins and an embedded multi-column legend matching your reference design."
)

if __name__ == "__main__":
    demo.queue().launch()
