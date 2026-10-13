import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
import os
import threading
import urllib.request
import glob
from PIL import Image, ImageTk

from uniface import RetinaFace
from models import XSeg

def setup_cuda_dlls():
    import sys
    paths_to_add = []
    for p in sys.path:
        if 'site-packages' in p:
            cudnn = os.path.join(p, 'nvidia', 'cudnn', 'bin')
            cublas = os.path.join(p, 'nvidia', 'cublas', 'bin')
            if os.path.exists(cudnn): paths_to_add.append(cudnn)
            if os.path.exists(cublas): paths_to_add.append(cublas)
            
    for path in paths_to_add:
        if hasattr(os, 'add_dll_directory'):
            try: os.add_dll_directory(path)
            except: pass
        os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')

setup_cuda_dlls()

DEFAULT_MODEL_PATH = "weights/xseg.onnx"

class FaceSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Segmentation - Offline GUI")
        self.root.geometry("1100x750")
        self.root.minsize(800, 600)
        
        # Models
        self.detector = None
        self.parser = None
        
        # State
        self.image_path = None
        self.original_image = None
        self.result_image = None
        self.mask_image = None
        self.stop_requested = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Left Panel (Controls) - use a canvas with scrollbar for overflow
        left_wrapper = tk.Frame(self.root, width=300)
        left_wrapper.pack(side=tk.LEFT, fill=tk.Y)
        
        left_canvas = tk.Canvas(left_wrapper, width=280, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_wrapper, orient=tk.VERTICAL, command=left_canvas.yview)
        control_frame = tk.Frame(left_canvas, padx=15, pady=15)
        
        control_frame.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=control_frame, anchor=tk.NW)
        left_canvas.configure(yscrollcommand=scrollbar.set)
        
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        tk.Label(control_frame, text="Model Path:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.model_path_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
        tk.Entry(control_frame, textvariable=self.model_path_var, width=35).pack(anchor=tk.W, pady=(0, 10))
        
        tk.Label(control_frame, text="Blur Sigma (Smoothing):", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.blur_var = tk.DoubleVar(value=0.0)
        tk.Scale(control_frame, variable=self.blur_var, from_=0, to=10, resolution=0.1, orient=tk.HORIZONTAL, length=200).pack(anchor=tk.W, pady=(0, 10))
        
        tk.Label(control_frame, text="Align Size:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.align_var = tk.IntVar(value=256)
        tk.Scale(control_frame, variable=self.align_var, from_=112, to=512, resolution=16, orient=tk.HORIZONTAL, length=200).pack(anchor=tk.W, pady=(0, 10))
        
        # --- Mask Controls ---
        sep1 = ttk.Separator(control_frame, orient=tk.HORIZONTAL)
        sep1.pack(fill=tk.X, pady=10)
        tk.Label(control_frame, text="Mask Controls:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.largest_face_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Largest face only", variable=self.largest_face_var).pack(anchor=tk.W)
        
        self.invert_mask_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="Invert mask", variable=self.invert_mask_var).pack(anchor=tk.W)
        
        tk.Label(control_frame, text="Mask Threshold:").pack(anchor=tk.W, pady=(5, 0))
        self.threshold_var = tk.DoubleVar(value=0.5)
        tk.Scale(control_frame, variable=self.threshold_var, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=200).pack(anchor=tk.W)
        
        tk.Label(control_frame, text="Erode / Dilate (px):").pack(anchor=tk.W, pady=(5, 0))
        self.erode_dilate_var = tk.IntVar(value=0)
        tk.Scale(control_frame, variable=self.erode_dilate_var, from_=-50, to=50, resolution=1, orient=tk.HORIZONTAL, length=200).pack(anchor=tk.W)
        
        tk.Label(control_frame, text="Feather (px):").pack(anchor=tk.W, pady=(5, 0))
        self.feather_var = tk.IntVar(value=0)
        tk.Scale(control_frame, variable=self.feather_var, from_=0, to=50, resolution=1, orient=tk.HORIZONTAL, length=200).pack(anchor=tk.W)
        
        tk.Label(control_frame, text="Mask Opacity:").pack(anchor=tk.W, pady=(5, 0))
        self.opacity_var = tk.DoubleVar(value=1.0)
        tk.Scale(control_frame, variable=self.opacity_var, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=200).pack(anchor=tk.W, pady=(0, 5))
        
        # --- Save Options ---
        sep2 = ttk.Separator(control_frame, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, pady=10)
        tk.Label(control_frame, text="Save Options:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.save_overlay_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="Save Overlay Result", variable=self.save_overlay_var).pack(anchor=tk.W)
        self.save_mask_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Save Segmentation Mask", variable=self.save_mask_var).pack(anchor=tk.W, pady=(0, 5))
        
        # Compute Device
        tk.Label(control_frame, text="Compute Device:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 0))
        self.use_cuda_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Use CUDA (GPU)", variable=self.use_cuda_var).pack(anchor=tk.W, pady=(0, 10))
        
        # --- Actions ---
        sep3 = ttk.Separator(control_frame, orient=tk.HORIZONTAL)
        sep3.pack(fill=tk.X, pady=10)
        tk.Label(control_frame, text="Single Image Mode", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.load_btn = tk.Button(control_frame, text="Load Image", command=self.load_image, width=25, height=2, bg="#e0e0e0")
        self.load_btn.pack(pady=2)
        
        self.process_btn = tk.Button(control_frame, text="Segment Face", command=self.process_image_thread, width=25, height=2, state=tk.DISABLED, bg="#d0e8f2")
        self.process_btn.pack(pady=2)
        
        self.save_btn = tk.Button(control_frame, text="Save Results", command=self.save_results, width=25, height=2, state=tk.DISABLED, bg="#d2f2d0")
        self.save_btn.pack(pady=(2, 15))
        
        tk.Label(control_frame, text="Batch Processing", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.batch_btn = tk.Button(control_frame, text="Batch Process Directory", command=self.batch_process_thread, width=25, height=2, bg="#f2e8d0")
        self.batch_btn.pack(pady=2)
        
        self.stop_btn = tk.Button(control_frame, text="Stop Processing", command=self.stop_processing, width=25, height=2, state=tk.DISABLED, bg="#ffcccc")
        self.stop_btn.pack(pady=2)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(10, 5))
        
        self.status_var = tk.StringVar(value="Ready. Please load an image or start batch.")
        tk.Label(control_frame, textvariable=self.status_var, fg="#333333", wraplength=250, justify=tk.LEFT).pack(pady=10, anchor=tk.SW)
        
        # Right Panel (Images)
        image_frame = tk.Frame(self.root, bg="#2b2b2b")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Top half: Input
        top_frame = tk.Frame(image_frame, bg="#2b2b2b")
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(top_frame, text="Input Image", bg="#2b2b2b", fg="white").pack()
        self.canvas_input = tk.Canvas(top_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas_input.pack(fill=tk.BOTH, expand=True)
        
        # Bottom half: Results side-by-side
        bottom_frame = tk.Frame(image_frame, bg="#2b2b2b")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        res_frame = tk.Frame(bottom_frame, bg="#2b2b2b")
        res_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(res_frame, text="Result Overlay", bg="#2b2b2b", fg="white").pack()
        self.canvas_result = tk.Canvas(res_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas_result.pack(fill=tk.BOTH, expand=True)
        
        mask_frame = tk.Frame(bottom_frame, bg="#2b2b2b")
        mask_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(mask_frame, text="Segmentation Mask", bg="#2b2b2b", fg="white").pack()
        self.canvas_mask = tk.Canvas(mask_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas_mask.pack(fill=tk.BOTH, expand=True)
        
        # Resize event
        self.root.bind("<Configure>", self.on_resize)
        
    def stop_processing(self):
        self.stop_requested = True
        self.status_var.set("Stopping... Please wait.")
        self.stop_btn.config(state=tk.DISABLED)
        
    def on_resize(self, event):
        if event.widget == self.root:
            if self.original_image is not None:
                self.display_image(self.original_image, self.canvas_input)
            if self.result_image is not None:
                self.display_image(self.result_image, self.canvas_result)
            if self.mask_image is not None:
                self.display_image(cv2.cvtColor(self.mask_image, cv2.COLOR_GRAY2BGR), self.canvas_mask)
                
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_path = path
            self.original_image = cv2.imread(path)
            if self.original_image is not None:
                self.process_btn.config(state=tk.NORMAL)
                self.display_image(self.original_image, self.canvas_input)
                self.canvas_result.delete("all")
                self.canvas_mask.delete("all")
                self.result_image = None
                self.mask_image = None
                self.save_btn.config(state=tk.DISABLED)
                self.status_var.set(f"Loaded {os.path.basename(path)}. Ready to segment.")
            else:
                messagebox.showerror("Error", "Failed to load image.")
                
    def display_image(self, cv_img, canvas):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        canvas.update_idletasks()
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w > 10 and h > 10:
            ratio = min(w/pil_img.width, h/pil_img.height)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            
        tk_img = ImageTk.PhotoImage(image=pil_img)
        canvas.delete("all")
        
        x = (w - tk_img.width()) // 2
        y = (h - tk_img.height()) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=tk_img)
        if canvas == self.canvas_input:
            self.tk_img_input = tk_img
        elif canvas == self.canvas_result:
            self.tk_img_result = tk_img
        elif canvas == self.canvas_mask:
            self.tk_img_mask = tk_img
            
    def apply_mask_visualization(self, image, mask, alpha=0.5):
        overlay = image.copy().astype(np.float32)
        mask_3ch = np.stack([mask * 0.3, mask * 0.7, mask * 0.3], axis=-1)
        overlay = overlay * (1 - mask[..., None] * alpha) + mask_3ch * 255 * alpha
        return overlay.clip(0, 255).astype(np.uint8)
        
    def download_weights_if_needed(self, model_path):
        if not os.path.exists(model_path):
            self.root.after(0, lambda: self.status_var.set("Downloading model weights... (This may take a moment)"))
            url = "https://github.com/yakhyo/face-segmentation/releases/download/weights/xseg.onnx"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
            self.root.after(0, lambda: self.status_var.set("Download complete!"))

    def init_models_if_needed(self):
        model_path = self.model_path_var.get()
        self.download_weights_if_needed(model_path)
            
        use_cuda = self.use_cuda_var.get()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
            
        if self.detector is None or getattr(self.detector, '_cuda_setting', None) != use_cuda:
            self.root.after(0, lambda: self.status_var.set("Initializing RetinaFace detector..."))
            try:
                self.detector = RetinaFace(providers=providers)
            except TypeError:
                self.detector = RetinaFace()
            self.detector._cuda_setting = use_cuda
            
        blur = self.blur_var.get()
        align = self.align_var.get()
        
        if self.parser is None or getattr(self.parser, 'model_path', None) != model_path or self.parser.align_size != align or self.parser.blur_sigma != blur or getattr(self.parser, '_cuda_setting', None) != use_cuda:
            self.root.after(0, lambda: self.status_var.set("Initializing XSeg model..."))
            self.parser = XSeg(model_path, align_size=align, blur_sigma=blur, providers=providers)
            self.parser.model_path = model_path
            self.parser._cuda_setting = use_cuda

    def postprocess_mask(self, mask):
        """Apply threshold, erode/dilate, feather, opacity, and invert to a float32 mask."""
        threshold = self.threshold_var.get()
        erode_dilate = self.erode_dilate_var.get()
        feather = self.feather_var.get()
        opacity = self.opacity_var.get()
        invert = self.invert_mask_var.get()
        
        # Threshold: binarize the soft mask
        mask = (mask >= threshold).astype(np.float32)
        
        # Erode (negative) / Dilate (positive)
        if erode_dilate != 0:
            kernel_size = abs(erode_dilate) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_uint8 = (mask * 255).astype(np.uint8)
            if erode_dilate < 0:
                mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
            else:
                mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
            mask = mask_uint8.astype(np.float32) / 255.0
        
        # Feather: Gaussian blur to soften edges
        if feather > 0:
            ksize = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        
        # Opacity
        mask = mask * opacity
        
        # Invert
        if invert:
            mask = 1.0 - mask
            
        return mask
            
    def core_segment_image(self, image):
        """Core segmentation function returning result_image and mask_image."""
        faces = self.detector.detect(image)
        if len(faces) == 0:
            return None, None
        
        # Filter to largest face only if enabled
        largest_only = self.largest_face_var.get()
        if largest_only:
            def face_area(face):
                x1, y1, x2, y2 = face.bbox[:4]
                return (x2 - x1) * (y2 - y1)
            faces = [max(faces, key=face_area)]
            
        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        for face in faces:
            landmarks = face.landmarks
            if landmarks is None or landmarks.shape[0] != 5:
                continue
            mask = self.parser.parse(image, landmarks)
            full_mask = np.maximum(full_mask, mask)
        
        # Apply mask post-processing
        full_mask = self.postprocess_mask(full_mask)
            
        result_image = self.apply_mask_visualization(image, full_mask)
        
        # Draw bboxes and landmarks
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox[:4])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if face.landmarks is not None:
                for lm in face.landmarks:
                    cv2.circle(result_image, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)
            
        mask_image = (full_mask * 255).clip(0, 255).astype(np.uint8)
        return result_image, mask_image

    def process_image_thread(self):
        self.process_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing... Please wait.")
        threading.Thread(target=self.process_image, daemon=True).start()
        
    def process_image(self):
        try:
            self.init_models_if_needed()
            self.root.after(0, lambda: self.status_var.set("Detecting and segmenting faces..."))
            
            image = self.original_image.copy()
            res_img, mask_img = self.core_segment_image(image)
            
            if res_img is None:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No faces detected in the image."))
                self.root.after(0, lambda: self.status_var.set("No faces detected."))
                return
                
            self.result_image = res_img
            self.mask_image = mask_img
            
            self.root.after(0, lambda: self.display_image(self.result_image, self.canvas_result))
            self.root.after(0, lambda: self.display_image(cv2.cvtColor(self.mask_image, cv2.COLOR_GRAY2BGR), self.canvas_mask))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("Processing complete!"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred:\n{str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error during processing."))
        finally:
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.load_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
            
    def save_results(self):
        if self.result_image is None or self.image_path is None:
            return
            
        if not self.save_overlay_var.get() and not self.save_mask_var.get():
            messagebox.showinfo("Info", "Both save options are disabled. Check at least one option to save.")
            return
            
        base_name_only = os.path.basename(self.image_path).rsplit(".", 1)[0]
        default_name = base_name_only + ("_result.png" if self.save_overlay_var.get() else "_mask.png")
        
        save_path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        
        if save_path:
            base_dir = os.path.dirname(save_path)
            chosen_name = os.path.basename(save_path).rsplit(".", 1)[0]
            if chosen_name.endswith("_result"): chosen_name = chosen_name[:-7]
            elif chosen_name.endswith("_mask"): chosen_name = chosen_name[:-5]
            
            saved_items = []
            if self.save_overlay_var.get():
                res_path = os.path.join(base_dir, f"{chosen_name}_result.png")
                cv2.imwrite(res_path, self.result_image)
                saved_items.append("Overlay Result")
                
            if self.save_mask_var.get():
                mask_path = os.path.join(base_dir, f"{chosen_name}_mask.png")
                cv2.imwrite(mask_path, self.mask_image)
                saved_items.append("Segmentation Mask")
                
            messagebox.showinfo("Success", f"Saved: {', '.join(saved_items)}\nTo: {base_dir}")

    def batch_process_thread(self):
        if not self.save_overlay_var.get() and not self.save_mask_var.get():
            messagebox.showinfo("Info", "Both save options are disabled. Check at least one option to save.")
            return
            
        input_dir = filedialog.askdirectory(title="Select Input Directory with Images")
        if not input_dir:
            return
            
        output_dir = filedialog.askdirectory(title="Select Output Directory for Results")
        if not output_dir:
            return
            
        self.stop_requested = False
        self.process_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)
        
        threading.Thread(target=self.batch_process, args=(input_dir, output_dir), daemon=True).start()
        
    def batch_process(self, input_dir, output_dir):
        try:
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
            image_files = []
            for f in os.listdir(input_dir):
                if f.lower().endswith(image_extensions):
                    image_files.append(os.path.join(input_dir, f))

                
            if not image_files:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No images found in the selected directory."))
                self.root.after(0, lambda: self.status_var.set("Batch processing cancelled (no images)."))
                return
                
            self.init_models_if_needed()
            
            total = len(image_files)
            success_count = 0
            
            save_overlay = self.save_overlay_var.get()
            save_mask = self.save_mask_var.get()
            
            for i, img_path in enumerate(image_files):
                if self.stop_requested:
                    self.root.after(0, lambda: messagebox.showinfo("Stopped", f"Batch processing was stopped by user.\nProcessed {i} out of {total} images.\nSaved to: {output_dir}"))
                    self.root.after(0, lambda: self.status_var.set("Batch processing stopped."))
                    return
                    
                self.root.after(0, lambda p=img_path, idx=i, t=total: self.status_var.set(f"Batch Processing {idx+1}/{t}: {os.path.basename(p)}"))
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                res_img, mask_img = self.core_segment_image(img)
                if res_img is not None:
                    original_name = os.path.basename(img_path)
                    base_name = original_name.rsplit(".", 1)[0]
                    
                    if save_overlay:
                        res_path = os.path.join(output_dir, f"{base_name}_result.png")
                        cv2.imwrite(res_path, res_img)
                        
                    if save_mask:
                        mask_path = os.path.join(output_dir, original_name)
                        cv2.imwrite(mask_path, mask_img)
                        
                    success_count += 1
                
                # Update progress
                progress_pct = ((i + 1) / total) * 100
                self.root.after(0, lambda pct=progress_pct: self.progress_var.set(pct))
                
            self.root.after(0, lambda: messagebox.showinfo("Batch Complete", f"Processed {total} images.\nSuccessfully segmented faces in {success_count} images.\nSaved to: {output_dir}"))
            self.root.after(0, lambda: self.status_var.set("Batch processing complete!"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Batch Error", f"An error occurred during batch processing:\n{str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Batch processing failed."))
        finally:
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.progress_var.set(0))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL if self.image_path else tk.DISABLED))
            self.root.after(0, lambda: self.load_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL if self.result_image is not None else tk.DISABLED))

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceSegmentationApp(root)
    root.mainloop()
