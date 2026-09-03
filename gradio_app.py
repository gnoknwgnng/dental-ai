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

print("Loading models for Gradio...")
anatomy_model = YOLO("models/anatomy_model.pt")
dental_model = YOLO("models/dental_model.pt")
print("Models loaded successfully!")

ANATOMY_PRIORITY_CLASSES = {"Maxillary sinus", "Mandibular Canal"}
CLASS_ALIASES = {
    "Inferior Alveolar Nerve Canal": "Mandibular Canal",
    "Maxillary Sinus": "Maxillary sinus"
}

# RGB Color palette for distinct visualization
COLORS = [
    (230, 25, 75),   # Red
    (60, 180, 75),   # Green
    (255, 225, 25),  # Yellow
    (67, 99, 216),   # Blue
    (245, 130, 49),  # Orange
    (145, 30, 180),  # Purple
    (70, 240, 240),  # Cyan
    (240, 50, 230),  # Magenta
    (188, 246, 12),  # Lime
    (250, 190, 190), # Pink
    (0, 128, 128),   # Teal
    (230, 190, 255), # Lavender
    (154, 99, 36),   # Brown
    (255, 250, 200), # Beige
    (128, 0, 0),     # Maroon
    (170, 255, 195)  # Mint
]

def normalize_name(name):
    return CLASS_ALIASES.get(name, name)

def extract_detections(result, names_dict, source):
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

        detections.append({
            "class_name": class_name,
            "confidence": confidence,
            "polygon": polygon,
            "source": source,
            "cx": cx,
            "cy": cy
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

    img_np = np.array(image.convert("RGB"))
    overlay = img_np.copy()
    h, w, _ = img_np.shape

    anatomy_dets = extract_detections(anatomy_result, anatomy_model.names, "Anatomical")
    dental_dets = extract_detections(dental_result, dental_model.names, "Dental")
    final_detections = merge_results(anatomy_dets, dental_dets)

    summary_rows = []

    for idx, det in enumerate(final_detections, start=1):
        det["id"] = idx
        color = COLORS[(idx - 1) % len(COLORS)]
        det["color"] = color

        pts = np.array(det["polygon"], dtype=np.int32)
        
        # 1. Draw polygon boundary contour
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

        # 2. Draw semi-transparent polygon fill
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        for c in range(3):
            overlay[:, :, c] = np.where(
                poly_mask == 255,
                img_np[:, :, c] * 0.65 + color[c] * 0.35,
                overlay[:, :, c]
            )

        # 3. Draw numbered badge & class label directly on image
        cx, cy = int(det["cx"]), int(det["cy"])

        # Badge circle background (white)
        cv2.circle(overlay, (cx, cy), 14, (255, 255, 255), -1)
        # Badge circle border (detection color)
        cv2.circle(overlay, (cx, cy), 14, color, 2)
        
        # Badge ID text inside circle
        id_text = str(idx)
        text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0]
        text_x = cx - text_size[0] // 2
        text_y = cy + text_size[1] // 2
        cv2.putText(overlay, id_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

        # Class Label Pill next to badge
        label = f" #{idx} {det['class_name']} ({det['confidence']:.0%})"
        lbl_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        lbl_x = cx + 18
        lbl_y = cy + 5

        # Ensure label fits within image bounds
        if lbl_x + lbl_size[0] + 6 > w:
            lbl_x = cx - 18 - lbl_size[0]

        # Label dark background box
        cv2.rectangle(overlay, (lbl_x - 2, lbl_y - lbl_size[1] - 4), (lbl_x + lbl_size[0] + 4, lbl_y + 4), (15, 23, 42), -1)
        # Label colored border line
        cv2.rectangle(overlay, (lbl_x - 2, lbl_y - lbl_size[1] - 4), (lbl_x + lbl_size[0] + 4, lbl_y + 4), color, 1)
        # Label text
        cv2.putText(overlay, label, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        conf_pct = f"{det['confidence']:.1%}"
        summary_rows.append(f"| **#{idx}** | **{det['class_name']}** | {det['source']} | `{conf_pct}` |")

    # Clean Markdown Table Summary & Legend
    if summary_rows:
        summary_text = (
            "### 📋 Detections Legend & Summary\n\n"
            "| ID Badge | Class Name | Category | Confidence |\n"
            "| :---: | :--- | :---: | :---: |\n" +
            "\n".join(summary_rows)
        )
    else:
        summary_text = "No detections found."

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return overlay, summary_text

# Define Gradio Interface
demo = gr.Interface(
    fn=predict_dental_xray,
    inputs=gr.Image(type="pil", label="Upload Dental X-Ray Image"),
    outputs=[
        gr.Image(type="numpy", label="Segmentation Output (Labeled with ID Badges)"),
        gr.Markdown(label="Detections Legend & Summary")
    ],
    title="🦷 Dental AI Segmentation Viewer",
    description="Upload a panoramic dental X-ray to perform dual AI instance segmentation for anatomical structures and dental findings. Each detected region is labeled with a numbered badge (#1, #2, #3...) matching the legend table below."
)

if __name__ == "__main__":
    demo.queue().launch()
