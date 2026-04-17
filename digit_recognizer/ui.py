import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tkinter as tk
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageTk
from tkinter import filedialog, messagebox, ttk

from digit_recognizer.config import (
    BRUSH_SIZE,
    CANVAS_SIZE,
    COLORS,
    COMPACT_CANVAS_SIZE,
    COMPACT_PREVIEW_SIZE,
    DEBUG_DIR,
    MODEL_PATH,
    PREVIEW_SIZE,
)
from digit_recognizer.model import load_saved_model, train_and_save_model


class DigitRecognizerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Nhan dang chu so viet tay - MNIST")
        self.root.geometry("980x680")
        self.root.minsize(760, 560)
        self.root.configure(bg=COLORS["bg"])

        self.model = None
        self.dataset_source = "chua san sang"
        self.last_processed_array: Optional[np.ndarray] = None
        self.last_source_path: Optional[Path] = None
        self.last_preview_image: Optional[Image.Image] = None
        self.last_preprocess_note = "none"
        self.is_training = False

        self.canvas_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        self.preview_photo = None

        # Khoi tao giao dien truoc, sau do moi load/train model de tranh khoa cua so.
        self._configure_styles()
        self._build_ui()
        self._bind_events()
        self._reset_preview()
        self._load_or_train_model_async(force_retrain=False)
        self.root.bind("<Configure>", self._on_window_resize)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        # Chia button thanh 3 nhom nhin ro hon: hanh dong chinh, phu, va debug/thoat.
        style.configure("Primary.TButton", font=("Bahnschrift SemiBold", 11), padding=(12, 10), background=COLORS["accent"], foreground="white", borderwidth=0)
        style.map("Primary.TButton", background=[("active", COLORS["accent_dark"]), ("disabled", "#d6d3d1")], foreground=[("disabled", "#f8fafc")])
        style.configure("Secondary.TButton", font=("Bahnschrift", 10), padding=(12, 9), background=COLORS["panel_alt"], foreground=COLORS["text"], borderwidth=1, relief="solid")
        style.map("Secondary.TButton", background=[("active", "#efe3d4"), ("disabled", "#ece7df")], foreground=[("disabled", "#9ca3af")], bordercolor=[("!disabled", COLORS["border"])])
        style.configure("Ghost.TButton", font=("Bahnschrift", 10), padding=(12, 9), background=COLORS["panel"], foreground=COLORS["accent_dark"], borderwidth=1, relief="solid")
        style.map("Ghost.TButton", background=[("active", COLORS["accent_soft"]), ("disabled", "#f1f5f9")], bordercolor=[("!disabled", COLORS["accent"])])

    def _build_ui(self) -> None:
        main_frame = tk.Frame(self.root, bg=COLORS["bg"], padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        content = tk.Frame(main_frame, bg=COLORS["bg"])
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=0)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        self.left_panel = tk.Frame(content, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1, padx=18, pady=18)
        self.left_panel.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        self.right_panel = tk.Frame(content, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1, padx=18, pady=18)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.content = content

        tk.Label(self.left_panel, text="Ve so bat ky", font=("Bahnschrift SemiBold", 14), bg=COLORS["panel"], fg=COLORS["text"]).pack(anchor="w")

        canvas_shell = tk.Frame(self.left_panel, bg=COLORS["panel_alt"], highlightbackground=COLORS["border"], highlightthickness=1, padx=10, pady=10)
        canvas_shell.pack()
        self.canvas = tk.Canvas(canvas_shell, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=COLORS["canvas_bg"], highlightthickness=2, highlightbackground=COLORS["accent_dark"], cursor="crosshair")
        self.canvas.pack()

        self.button_frame = tk.Frame(self.left_panel, bg=COLORS["panel"])
        self.button_frame.pack(fill="x", pady=(16, 0))
        self.predict_button = ttk.Button(self.button_frame, text="Du doan", style="Primary.TButton", command=self.predict_from_canvas)
        self.clear_button = ttk.Button(self.button_frame, text="Xoa", style="Secondary.TButton", command=self.clear_canvas)
        self.upload_button = ttk.Button(self.button_frame, text="Tai anh", style="Secondary.TButton", command=self.upload_image)
        self.debug_button = ttk.Button(self.button_frame, text="Luu debug", style="Ghost.TButton", command=self.save_debug_snapshot)
        self.retrain_button = ttk.Button(self.button_frame, text="Load lai model", style="Secondary.TButton", command=lambda: self._load_or_train_model_async(force_retrain=True))
        self.exit_button = ttk.Button(self.button_frame, text="Thoat", style="Ghost.TButton", command=self.root.destroy)
        self._layout_buttons(compact=False)

        self.status_var = tk.StringVar(value="Dang khoi tao model...")
        status_card = tk.Frame(self.left_panel, bg=COLORS["accent_soft"], highlightbackground=COLORS["border"], highlightthickness=1, padx=10, pady=10)
        status_card.pack(fill="x", pady=(16, 0))
        tk.Label(status_card, textvariable=self.status_var, font=("Bahnschrift", 10), justify="left", wraplength=280, bg=COLORS["accent_soft"], fg=COLORS["accent_dark"]).pack(anchor="w")

        tk.Label(self.right_panel, text="Ket qua du doan", font=("Bahnschrift SemiBold", 14), bg=COLORS["panel"], fg=COLORS["text"]).grid(row=0, column=0, sticky="w")
        prediction_card = tk.Frame(self.right_panel, bg=COLORS["panel_alt"], highlightbackground=COLORS["border"], highlightthickness=1, padx=18, pady=14)
        prediction_card.grid(row=1, column=0, sticky="ew", pady=(8, 16))
        prediction_card.grid_columnconfigure(1, weight=1)
        tk.Label(prediction_card, text="So du doan", font=("Bahnschrift", 10), bg=COLORS["panel_alt"], fg=COLORS["muted"]).grid(row=0, column=0, sticky="w")
        self.dataset_badge_var = tk.StringVar(value="Dataset: dang khoi tao...")
        tk.Label(prediction_card, textvariable=self.dataset_badge_var, font=("Bahnschrift SemiBold", 9), bg=COLORS["accent_soft"], fg=COLORS["accent_dark"], padx=10, pady=5).grid(row=0, column=1, sticky="e")
        self.result_var = tk.StringVar(value="-")
        tk.Label(prediction_card, textvariable=self.result_var, font=("Cambria", 44, "bold"), bg=COLORS["panel_alt"], fg=COLORS["accent"]).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        tk.Label(self.right_panel, text="Anh 28x28 sau tien xu ly", font=("Bahnschrift SemiBold", 12), bg=COLORS["panel"], fg=COLORS["text"]).grid(row=2, column=0, sticky="w")
        self.preview_label = tk.Label(self.right_panel, width=PREVIEW_SIZE, height=PREVIEW_SIZE, bg=COLORS["preview_bg"], bd=0, highlightbackground=COLORS["border"], highlightthickness=1)
        self.preview_label.grid(row=3, column=0, sticky="w", pady=(10, 16))
        self.current_preview_size = PREVIEW_SIZE

        tk.Label(self.right_panel, text="Top xac suat", font=("Bahnschrift SemiBold", 12), bg=COLORS["panel"], fg=COLORS["text"]).grid(row=4, column=0, sticky="w")
        self.probabilities_text = tk.Text(self.right_panel, height=8, width=42, font=("Consolas", 11), bg="#fffdf9", fg=COLORS["text"], bd=1, relief="flat", highlightbackground=COLORS["border"], highlightthickness=1, padx=10, pady=10)
        self.probabilities_text.grid(row=5, column=0, sticky="ew")
        self.probabilities_text.insert("1.0", "Chua co du doan.\n")
        self.probabilities_text.config(state="disabled")

        self.source_var = tk.StringVar(value="Nguon hien tai: Canvas")
        tk.Label(self.right_panel, textvariable=self.source_var, font=("Bahnschrift", 10), bg=COLORS["panel"], fg=COLORS["muted"]).grid(row=6, column=0, sticky="w", pady=(16, 0))

    def _on_window_resize(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        # Chuyen sang layout compact khi cua so nho de tranh vo bo cuc.
        compact = event.width < 860 or event.height < 620
        self.left_panel.grid_configure(padx=(0, 0 if compact else 16), pady=(0, 12 if compact else 0))
        if compact:
            self.content.grid_columnconfigure(0, weight=1)
            self.content.grid_columnconfigure(1, weight=1)
            self.left_panel.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0)
            self.right_panel.grid(row=1, column=0, columnspan=2, sticky="nsew")
        else:
            self.content.grid_columnconfigure(0, weight=0)
            self.content.grid_columnconfigure(1, weight=1)
            self.left_panel.grid(row=0, column=0, columnspan=1, sticky="nsw", padx=(0, 16), pady=0)
            self.right_panel.grid(row=0, column=1, columnspan=1, sticky="nsew")
        target_canvas = COMPACT_CANVAS_SIZE if compact else CANVAS_SIZE
        if int(self.canvas.cget("width")) != target_canvas:
            self.canvas.config(width=target_canvas, height=target_canvas)
        target_preview = COMPACT_PREVIEW_SIZE if compact else PREVIEW_SIZE
        if self.current_preview_size != target_preview:
            self.current_preview_size = target_preview
            if self.last_preview_image is not None:
                self._update_preview(self.last_preview_image)
        self._layout_buttons(compact)

    def _layout_buttons(self, compact: bool) -> None:
        for widget in self.button_frame.winfo_children():
            widget.pack_forget()
        if compact:
            self.predict_button.pack(fill="x", pady=3)
            row_one = tk.Frame(self.button_frame, bg=COLORS["panel"])
            row_one.pack(fill="x", pady=3)
            self.clear_button.pack(in_=row_one, side="left", fill="x", expand=True, padx=(0, 4))
            self.upload_button.pack(in_=row_one, side="left", fill="x", expand=True, padx=(4, 0))
            row_two = tk.Frame(self.button_frame, bg=COLORS["panel"])
            row_two.pack(fill="x", pady=3)
            self.debug_button.pack(in_=row_two, side="left", fill="x", expand=True, padx=(0, 4))
            self.retrain_button.pack(in_=row_two, side="left", fill="x", expand=True, padx=(4, 0))
            self.exit_button.pack(fill="x", pady=3)
            return
        self.predict_button.pack(fill="x", pady=4)
        self.clear_button.pack(fill="x", pady=4)
        self.upload_button.pack(fill="x", pady=4)
        self.debug_button.pack(fill="x", pady=4)
        self.retrain_button.pack(fill="x", pady=4)
        self.exit_button.pack(fill="x", pady=4)

    def _bind_events(self) -> None:
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<Button-1>", self._paint)

    def _paint(self, event: tk.Event) -> None:
        x1 = event.x - BRUSH_SIZE
        y1 = event.y - BRUSH_SIZE
        x2 = event.x + BRUSH_SIZE
        y2 = event.y + BRUSH_SIZE
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.canvas_draw.ellipse((x1, y1, x2, y2), fill=255)
        # Moi khi nguoi dung ve lai thi bo ket qua cu de tranh nham lan.
        self.last_source_path = None
        self.last_processed_array = None
        self.result_var.set("-")
        self._set_probabilities_text("Dang dung du lieu tu canvas. Nhan 'Du doan' sau khi ve xong.\n")
        self.source_var.set("Nguon hien tai: Canvas")

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.canvas_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        self.last_processed_array = None
        self.last_source_path = None
        self.last_preprocess_note = "none"
        self.result_var.set("-")
        self.source_var.set("Nguon hien tai: Canvas")
        self._set_probabilities_text("Chua co du doan.\n")
        self._reset_preview()

    def upload_image(self) -> None:
        file_path = filedialog.askopenfilename(title="Chon anh chu so", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            with Image.open(file_path) as uploaded:
                processed, preview = self.preprocess_image(uploaded.convert("L"))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Loi anh", f"Khong the mo hoac xu ly anh da chon.\nChi tiet: {exc}")
            return
        self.last_processed_array = processed
        self.last_source_path = Path(file_path)
        self.last_preview_image = preview.copy()
        self.result_var.set("-")
        self.source_var.set(f"Nguon hien tai: {self.last_source_path.name}")
        self._update_preview(preview)
        self._set_probabilities_text("Anh da duoc tai len va xu ly.\nNhan 'Du doan' de nhan ket qua.\n")

    def predict_from_canvas(self) -> None:
        if self.model is None:
            messagebox.showinfo("Model chua san sang", "Model dang duoc huan luyen hoac tai len. Vui long doi them.")
            return
        if self.last_source_path is None:
            if not self._canvas_has_drawing():
                messagebox.showwarning("Canvas trong", "Ban chua ve gi tren canvas. Hay ve mot chu so roi thu lai.")
                return
            processed, preview = self.preprocess_image(self.canvas_image)
            self.last_processed_array = processed
            self.last_preview_image = preview.copy()
            self._update_preview(preview)
        if self.last_processed_array is None:
            messagebox.showwarning("Khong co du lieu", "Khong tim thay du lieu anh de du doan.")
            return
        self._predict_with_array(self.last_processed_array)

    def _predict_with_array(self, processed_array: np.ndarray) -> None:
        sample = processed_array.reshape(1, -1)
        prediction = int(self.model.predict(sample)[0])
        self.result_var.set(str(prediction))
        debug_lines = [f"Dataset: {self.dataset_source}"]
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(sample)[0]
            top_indices = np.argsort(probabilities)[::-1][:5]
            debug_lines.extend([f"So {idx}: {probabilities[idx] * 100:6.2f}%" for idx in top_indices])
            debug_lines.append(f"Preprocess: {self.last_preprocess_note}")
            debug_lines.append(f"Pixel mean: {processed_array.mean():.4f}")
            debug_lines.append(f"Pixel max:  {processed_array.max():.4f}")
            debug_lines.append(f"Pixel min:  {processed_array.min():.4f}")
        else:
            debug_lines.append("Model hien tai khong ho tro predict_proba.")
        self._set_probabilities_text("\n".join(debug_lines) + "\n")

    def preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, Image.Image]:
        grayscale = image.convert("L")
        # Canvas dung nen den net trang, nhung anh upload co the nguoc lai.
        if np.array(grayscale).mean() > 127:
            grayscale = ImageOps.invert(grayscale)
        grayscale = ImageOps.autocontrast(grayscale)
        # Cat sat chu so de resize khong bi phi nhieu dien tich vao phan nen.
        bbox = grayscale.point(lambda p: 255 if p > 20 else 0).getbbox()
        if bbox is None:
            raise ValueError("Khong phat hien duoc chu so hop le trong anh.")
        cropped = grayscale.crop(bbox)
        centered = self._fit_and_center_digit(cropped)
        centered, preprocess_note = self._refine_for_mnist_style(centered)
        self.last_preprocess_note = preprocess_note
        normalized = np.asarray(centered, dtype=np.float32) / 255.0
        return normalized.reshape(-1), centered

    def _fit_and_center_digit(self, digit_image: Image.Image) -> Image.Image:
        max_side = max(digit_image.size)
        scale = 20.0 / max_side if max_side else 1.0
        new_size = (max(1, int(round(digit_image.width * scale))), max(1, int(round(digit_image.height * scale))))
        resized = digit_image.resize(new_size, Image.Resampling.LANCZOS)
        background = Image.new("L", (28, 28), color=0)
        background.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
        array = np.asarray(background, dtype=np.float32)
        if np.sum(array) > 0:
            # Dua tam khoi luong ve giua anh de giong cach can giua trong MNIST.
            cy, cx = self._center_of_mass(array)
            background = ImageChops.offset(background, int(round(13.5 - cx)), int(round(13.5 - cy)))
        return background

    def _refine_for_mnist_style(self, image_28: Image.Image) -> Tuple[Image.Image, str]:
        array = np.asarray(image_28, dtype=np.float32) / 255.0
        ys, xs = np.where(array > 0.2)
        if len(xs) == 0:
            return image_28, "empty"
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)
        active_pixels = int((array > 0.2).sum())
        # So 1 viet tay thuong qua day, de bi nham sang 8/9; lam mong nhe truong hop nay.
        if bbox_h >= 18 and bbox_w <= 10 and active_pixels >= 95:
            return image_28.filter(ImageFilter.MinFilter(3)), "thin_narrow_digit"
        return image_28, "standard"

    @staticmethod
    def _center_of_mass(array: np.ndarray) -> Tuple[float, float]:
        total = array.sum()
        if total == 0:
            return 13.5, 13.5
        y_indices, x_indices = np.indices(array.shape)
        return float((y_indices * array).sum() / total), float((x_indices * array).sum() / total)

    def _canvas_has_drawing(self) -> bool:
        return self.canvas_image.getbbox() is not None

    def _update_preview(self, image_28: Image.Image) -> None:
        self.last_preview_image = image_28.copy()
        preview = image_28.resize((self.current_preview_size, self.current_preview_size), Image.Resampling.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_photo)

    def _reset_preview(self) -> None:
        self._update_preview(Image.new("L", (28, 28), color=0))

    def save_debug_snapshot(self) -> None:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            canvas_path = DEBUG_DIR / "latest_canvas.png"
            processed_path = DEBUG_DIR / "latest_processed_28x28.png"
            info_path = DEBUG_DIR / "debug_info.txt"
            self.canvas_image.save(canvas_path)
            if self.last_preview_image is not None:
                self.last_preview_image.save(processed_path)
            # Luu them metadata de so sanh prediction voi anh sau preprocess.
            info_lines = [
                f"dataset_source={self.dataset_source}",
                f"model_path={MODEL_PATH.resolve()}",
                f"source_path={self.last_source_path if self.last_source_path else 'canvas'}",
                f"prediction={self.result_var.get()}",
                f"preprocess_note={self.last_preprocess_note}",
            ]
            if self.last_processed_array is not None:
                info_lines.extend([
                    f"processed_mean={float(self.last_processed_array.mean()):.6f}",
                    f"processed_max={float(self.last_processed_array.max()):.6f}",
                    f"processed_min={float(self.last_processed_array.min()):.6f}",
                ])
            info_path.write_text("\n".join(info_lines), encoding="utf-8")
            messagebox.showinfo("Da luu debug", f"Da luu thong tin debug vao thu muc:\n{DEBUG_DIR.resolve()}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Loi debug", f"Khong the luu du lieu debug.\nChi tiet: {exc}")

    def _set_probabilities_text(self, content: str) -> None:
        self.probabilities_text.config(state="normal")
        self.probabilities_text.delete("1.0", tk.END)
        self.probabilities_text.insert("1.0", content)
        self.probabilities_text.config(state="disabled")

    def _set_ui_busy(self, busy: bool, message: str) -> None:
        self.is_training = busy
        self.predict_button.config(state="normal")
        self.retrain_button.config(state="disabled" if busy else "normal")
        self.status_var.set(message)
        if busy:
            self.dataset_badge_var.set("Dataset: dang khoi tao")

    def _load_or_train_model_async(self, force_retrain: bool) -> None:
        if self.is_training:
            return
        self._set_ui_busy(True, "Dang tai model da luu..." if MODEL_PATH.exists() and not force_retrain else "Dang huan luyen model tren MNIST. Lan dau co the mat vai phut...")
        # Dung thread rieng de GUI van responsive trong luc load/train.
        threading.Thread(target=self._load_or_train_model_worker, args=(force_retrain,), daemon=True).start()

    def _load_or_train_model_worker(self, force_retrain: bool) -> None:
        try:
            if MODEL_PATH.exists() and not force_retrain:
                model, dataset_source = load_saved_model()
                # Quay lai main thread de cap nhat widget an toan.
                self.root.after(0, lambda: self._on_model_ready(model, f"Da tai model tu file luu. Nguon du lieu: {dataset_source}", dataset_source))
                return
            model, accuracy, report, confusion_path, dataset_source = train_and_save_model()
            print(f"\nTest accuracy: {accuracy:.4f}")
            print("\nClassification report:")
            print(report)
            self.root.after(0, lambda: self._on_model_ready(model, f"Da huan luyen xong model tu {dataset_source}. Test accuracy = {accuracy:.4f}", dataset_source, confusion_path))
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda: self._on_model_error(exc))

    def _on_model_ready(self, model, message: str, dataset_source: str, confusion_path: Optional[Path] = None) -> None:
        self.model = model
        self.dataset_source = dataset_source
        self.dataset_badge_var.set(f"Dataset: {dataset_source}")
        self._set_ui_busy(False, message)
        self._set_probabilities_text(f"Model san sang. Nguon du lieu: {dataset_source}.\nHay ve so hoac tai anh de bat dau.\n")
        if confusion_path and confusion_path.exists():
            self._show_confusion_matrix_window(confusion_path)

    def _on_model_error(self, exc: Exception) -> None:
        self._set_ui_busy(False, "Khong the khoi tao model.")
        self.dataset_badge_var.set("Dataset: loi khoi tao")
        self._set_probabilities_text("Model chua san sang do loi khoi tao.\n")
        messagebox.showerror("Loi model", "Khong the tai hoac huan luyen model.\nApp uu tien MNIST, nhung neu he thong offline thi se fallback sang digits.\n\n" + f"Chi tiet: {exc}")

    def _show_confusion_matrix_window(self, image_path: Path) -> None:
        try:
            with Image.open(image_path) as image:
                display_image = image.copy()
        except Exception:
            return
        display_image.thumbnail((520, 520), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)
        window = tk.Toplevel(self.root)
        window.title("Confusion Matrix")
        window.configure(bg="white")
        window.geometry(f"{photo.width() + 30}x{photo.height() + 50}")
        label = tk.Label(window, image=photo, bg="white")
        label.image = photo
        label.pack(padx=15, pady=15)


def main() -> None:
    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()
