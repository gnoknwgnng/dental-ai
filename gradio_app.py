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

COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (67, 99, 216),
    (245, 130, 49), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (188, 246, 12), (250, 190, 190), (0, 128, 128), (230, 190, 255)
]

@spaces.GPU
def predict_dental_xray(image):
    if image is None:
        return None, "Please upload an image."

    # Convert to PIL/RGB if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Temporary save for YOLO prediction in /tmp
    temp_path = "/tmp/temp_input.jpg"
    image.save(temp_path)

    # Run inference
    anatomy_result = anatomy_model.predict(source=temp_path, imgsz=1024, conf=0.25, verbose=False)[0]
    dental_result = dental_model.predict(source=temp_path, imgsz=1024, conf=0.25, verbose=False)[0]

    img_np = np.array(image.convert("RGB"))
    overlay = img_np.copy()
    h, w, _ = img_np.shape

    summary = []

    def draw_masks(result, source_name):
        if result.masks is None:
            return
        for idx, (mask, box) in enumerate(zip(result.masks.xy, result.boxes), start=1):
            cls_id = int(box.cls[0])
            raw_name = result.names[cls_id]
            cls_name = CLASS_ALIASES.get(raw_name, raw_name)
            conf = float(box.conf[0])

            color = COLORS[idx % len(COLORS)]

            pts = np.array(mask, dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)
            
            # Semi-transparent fill
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [pts], 255)
            for c in range(3):
                overlay[:, :, c] = np.where(poly_mask == 255, 
                                            img_np[:, :, c] * 0.6 + color[c] * 0.4, 
                                            overlay[:, :, c])

            summary.append(f"• **{cls_name}** ({source_name.title()}) - Confidence: {conf:.1%}")

    draw_masks(anatomy_result, "Anatomical")
    draw_masks(dental_result, "Dental")

    summary_text = "\n".join(summary) if summary else "No detections found."

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
        gr.Image(type="numpy", label="Segmentation Output"),
        gr.Markdown(label="Detections Summary")
    ],
    title="🦷 Dental AI Segmentation Viewer",
    description="Upload a panoramic dental X-ray to perform dual AI instance segmentation for anatomical structures and dental findings."
)

if __name__ == "__main__":
    demo.queue().launch()
