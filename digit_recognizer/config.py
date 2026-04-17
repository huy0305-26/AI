from pathlib import Path


MODEL_PATH = Path("mnist_model.pkl")
DATA_HOME = Path("data_cache")
CONFUSION_MATRIX_PATH = Path("confusion_matrix.png")
DEBUG_DIR = Path("debug_output")

MODEL_TYPE = "logistic_regression"
CANVAS_SIZE = 280
PREVIEW_SIZE = 140
COMPACT_CANVAS_SIZE = 220
COMPACT_PREVIEW_SIZE = 110
BRUSH_SIZE = 18
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_TRAIN_SAMPLES = 20000

COLORS = {
    "bg": "#f5efe6",
    "panel": "#fffaf2",
    "panel_alt": "#f8f1e7",
    "text": "#1f2937",
    "muted": "#6b7280",
    "border": "#d6c7b6",
    "accent": "#c2410c",
    "accent_dark": "#9a3412",
    "accent_soft": "#fde6d8",
    "success": "#166534",
    "canvas_bg": "#140f0a",
    "preview_bg": "#171717",
}

