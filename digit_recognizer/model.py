import warnings
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from sklearn.datasets import fetch_openml, load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from digit_recognizer.config import (
    CONFUSION_MATRIX_PATH,
    DATA_HOME,
    MAX_TRAIN_SAMPLES,
    MODEL_PATH,
    MODEL_TYPE,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_saved_model():
    # File model duoc luu theo dang dict de co them metadata ve nguon dataset.
    model_bundle = joblib.load(MODEL_PATH)
    if isinstance(model_bundle, dict):
        model = model_bundle["model"]
        dataset_source = model_bundle.get("dataset_source", "MNIST")
    else:
        model = model_bundle
        dataset_source = "MNIST"
    return model, dataset_source


def train_and_save_model():
    # Uu tien train tu MNIST, neu that bai moi fallback sang digits.
    X, y, dataset_source = load_training_data()

    if MAX_TRAIN_SAMPLES and len(X) > MAX_TRAIN_SAMPLES:
        # Cat mau de thoi gian train hop ly hon tren may demo.
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=MAX_TRAIN_SAMPLES,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if MODEL_TYPE == "decision_tree":
        model = DecisionTreeClassifier(max_depth=30, random_state=RANDOM_STATE)
    else:
        model = LogisticRegression(solver="lbfgs", max_iter=400, verbose=0)

    with warnings.catch_warnings():
        # Chan warning hoi tu de terminal gon hon trong qua trinh demo.
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    save_confusion_matrix(y_test, y_pred, dataset_source, CONFUSION_MATRIX_PATH)

    bundle = {
        "model": model,
        "model_type": MODEL_TYPE,
        "classes": list(getattr(model, "classes_", [])),
        "dataset_source": dataset_source,
    }
    joblib.dump(bundle, MODEL_PATH)
    return model, accuracy, report, CONFUSION_MATRIX_PATH, dataset_source


def load_training_data() -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        X, y = load_mnist()
        return X, y, "MNIST"
    except Exception as exc:  # noqa: BLE001
        # Fallback giup app van chay duoc neu may khong tai duoc OpenML.
        print(f"Khong the tai MNIST, chuyen sang digits fallback. Chi tiet: {exc}")
        X, y = load_digits_fallback()
        return X, y, "sklearn digits fallback"


def load_mnist() -> Tuple[np.ndarray, np.ndarray]:
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # parser='liac-arff' tranh phu thuoc vao pandas trong moi truong nhe.
        dataset = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            data_home=str(DATA_HOME.resolve()),
            parser="liac-arff",
        )
    X = dataset.data.astype(np.float32) / 255.0
    y = dataset.target.astype(np.int64)
    return X, y


def load_digits_fallback() -> Tuple[np.ndarray, np.ndarray]:
    digits = load_digits()
    resized_images = []
    for image in digits.images:
        # digits goc la 8x8, can phong to len 28x28 de dong nhat voi pipeline MNIST.
        pil_image = Image.fromarray(np.uint8((image / 16.0) * 255), mode="L")
        pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
        resized_images.append(np.asarray(pil_image, dtype=np.float32) / 255.0)

    X = np.stack(resized_images).reshape(len(resized_images), -1)
    y = digits.target.astype(np.int64)
    return X, y


def save_confusion_matrix(y_true, y_pred, dataset_source: str, output_path: Path) -> None:
    # Render confusion matrix ra file anh de co the mo bang Tkinter ma khong phu thuoc backend GUI cua matplotlib.
    figure = Figure(figsize=(6, 6), dpi=120)
    axis = figure.add_subplot(111)
    confusion_display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        values_format="d",
        ax=axis,
    )
    confusion_display.ax_.set_title(f"Confusion Matrix - {dataset_source}")
    figure.tight_layout()
    FigureCanvasAgg(figure).draw()
    figure.savefig(output_path)
