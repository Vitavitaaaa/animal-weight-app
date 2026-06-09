from pathlib import Path

import torch

from backend.models.classifier import get_classifier
from backend.models.segmentation import UNet
from backend.models.regression import TwoStreamRegressor

from backend.utils.preprocess import (
    load_image,
    preprocess_classifier,
    preprocess_segmentation,
    preprocess_regression
)

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

WEIGHTS_DIR = BASE_DIR / "weights"

# =========================================================
# LOAD CLASSIFIER
# =========================================================

classifier_model = get_classifier().to(DEVICE)

classifier_checkpoint = torch.load(
    WEIGHTS_DIR / "cattle_classifier.pth",
    map_location=DEVICE,
    weights_only=False
)

classifier_model.load_state_dict(
    classifier_checkpoint["model_state_dict"]
)

classifier_model.eval()

# =========================================================
# LOAD SEGMENTATION
# =========================================================

segmentation_model = UNet().to(DEVICE)

segmentation_checkpoint = torch.load(
    WEIGHTS_DIR / "cattle_segmentation.pth",
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" in segmentation_checkpoint:

    segmentation_model.load_state_dict(
        segmentation_checkpoint["model_state_dict"]
    )

else:

    segmentation_model.load_state_dict(
        segmentation_checkpoint
    )

segmentation_model.eval()

# =========================================================
# LOAD REGRESSION
# =========================================================

regression_model = TwoStreamRegressor(
    segmentation_model
).to(DEVICE)

regression_checkpoint = torch.load(
    WEIGHTS_DIR / "cattle_regressor.pth",
    map_location=DEVICE,
    weights_only=False
)

regression_model.load_state_dict(
    regression_checkpoint
)

regression_model.eval()

# =========================================================
# CONFIG
# =========================================================

CLASSIFICATION_THRESHOLD = 0.5

MAPE_SCORE = 13.40

# =========================================================
# PIPELINE
# =========================================================

def run_pipeline(image_path):

    image = load_image(image_path)

    # =====================================================
    # CLASSIFIER
    # =====================================================

    cls_tensor = preprocess_classifier(image).to(DEVICE)

    with torch.no_grad():

        cls_output = classifier_model(cls_tensor)

        cls_prob = torch.sigmoid(
            cls_output
        ).item()

    is_cow = cls_prob > CLASSIFICATION_THRESHOLD

    if not is_cow:

        return {
            "success": True,
            "is_cow": False,
            "message": "No cow detected in the image"
        }

    # =====================================================
    # SEGMENTATION
    # =====================================================

    seg_tensor = preprocess_segmentation(image).to(DEVICE)

    with torch.no_grad():

        seg_logits = segmentation_model(seg_tensor)

        seg_mask = torch.argmax(
            seg_logits,
            dim=1
        )

    # =====================================================
    # REGRESSION
    # =====================================================

    reg_tensor = preprocess_regression(image).to(DEVICE)

    with torch.no_grad():

        pred_weight = regression_model(
            reg_tensor
        )

    pred_weight = round(
        pred_weight.item(),
        2
    )

    return {
        "success": True,
        "is_cow": True,
        "predicted_weight": pred_weight,
        "mape": MAPE_SCORE,
        "classifier_confidence": round(
            cls_prob * 100,
            2
        )
    }
