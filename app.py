from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    render_template
)

from ultralytics import YOLO

import numpy as np
import os
import uuid

# ==========================================================
# CONFIG
# ==========================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================================================
# LOAD MODELS
# ==========================================================

print("Loading models...")

anatomy_model = YOLO("models/anatomy_model.pt")
dental_model = YOLO("models/dental_model.pt")

print("Models loaded!")

# ==========================================================
# CLASS PRIORITY
# ==========================================================

ANATOMY_PRIORITY_CLASSES = {
    "Maxillary sinus",
    "Mandibular Canal"
}

CLASS_ALIASES = {
    "Inferior Alveolar Nerve Canal": "Mandibular Canal",
    "Maxillary Sinus": "Maxillary sinus"
}

# ==========================================================
# COLORS
# ==========================================================

COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080"
]

# ==========================================================
# HELPERS
# ==========================================================

def normalize_name(name):

    return CLASS_ALIASES.get(name, name)


def extract_detections(
    result,
    names_dict,
    source,
    image_width,
    image_height
):

    detections = []

    if result.masks is None:
        return detections

    for mask, box in zip(
        result.masks.xy,
        result.boxes
    ):

        cls_id = int(box.cls[0])

        class_name = names_dict[cls_id]

        class_name = normalize_name(
            class_name
        )

        confidence = float(
            box.conf[0]
        )

        polygon = (
            np.array(mask)
            .astype(int)
            .tolist()
        )

        poly_np = np.array(polygon)

        cx = float(
            np.mean(poly_np[:, 0])
        )

        cy = float(
            np.mean(poly_np[:, 1])
        )

        d_left = cx
        d_right = image_width - cx
        d_top = cy
        d_bottom = image_height - cy

        side = int(
            np.argmin(
                [
                    d_left,
                    d_right,
                    d_top,
                    d_bottom
                ]
            )
        )

        detections.append({

            "class_name": class_name,

            "confidence": round(
                confidence,
                4
            ),

            "polygon": polygon,

            "source": source,

            "cx": cx,

            "cy": cy,

            "side": side
        })

    return detections


def merge_results(
    anatomy_dets,
    dental_dets
):

    merged = []

    merged.extend(
        anatomy_dets
    )

    for det in dental_dets:

        if (
            det["class_name"]
            in ANATOMY_PRIORITY_CLASSES
        ):
            continue

        merged.append(det)

    return merged


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# PREDICT
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return jsonify(
            {
                "error":
                "No image uploaded"
            }
        ), 400

    file = request.files["image"]

    ext = os.path.splitext(
        file.filename
    )[1]

    filename = (
        f"{uuid.uuid4()}{ext}"
    )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(image_path)

    # ======================================================
    # INFERENCE
    # ======================================================

    anatomy_result = anatomy_model.predict(
        source=image_path,
        imgsz=1024,
        conf=0.25,
        iou=0.5,
        retina_masks=True,
        save=False,
        verbose=False
    )[0]

    dental_result = dental_model.predict(
        source=image_path,
        imgsz=1024,
        conf=0.25,
        verbose=False
    )[0]

    H, W = anatomy_result.orig_shape

    # ======================================================
    # EXTRACT
    # ======================================================

    anatomy_dets = extract_detections(
        anatomy_result,
        anatomy_model.names,
        "anatomy",
        W,
        H
    )

    dental_dets = extract_detections(
        dental_result,
        dental_model.names,
        "dental",
        W,
        H
    )

    final_detections = merge_results(
        anatomy_dets,
        dental_dets
    )

    # ======================================================
    # IDS + COLORS
    # ======================================================

    for idx, det in enumerate(
        final_detections,
        start=1
    ):

        det["id"] = idx

        det["color"] = COLORS[
            (idx - 1)
            % len(COLORS)
        ]

    # ======================================================
    # RESPONSE
    # ======================================================

    return jsonify({

        "image_url":
        f"/uploads/{filename}",

        "width":
        int(W),

        "height":
        int(H),

        "detections":
        final_detections
    })


# ==========================================================
# UPLOADS
# ==========================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )