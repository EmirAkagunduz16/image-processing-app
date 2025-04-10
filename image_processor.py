import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, Menu
from tkinter import ttk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import os

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Uygulaması - Orijinal Görüntü")
        
        # Create second window for processed image
        self.processed_window = tk.Toplevel(root)
        self.processed_window.title("Görüntü İşleme Uygulaması - İşlenmiş Görüntü")
        self.processed_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
        # Original and processed images
        self.original_image = None
        self.processed_image = None
        self.current_image = None  # Track current state of image
        self.rotation_angle = 0  # Track rotation angle
        
        # Add threshold method variable
        self.threshold_method = tk.StringVar(value='binary')
        
        self.setup_menu()
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_image())
        self.root.bind('<Control-s>', lambda e: self.save_image())
        self.root.bind('<Control-z>', lambda e: self.reset_image())
    
    def setup_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Dosya", menu=file_menu)
        file_menu.add_command(label="Görüntü Yükle (Ctrl+O)", command=self.load_image)
        file_menu.add_command(label="Görüntü Kaydet (Ctrl+S)", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Çıkış", command=self.root.quit)
        
        # Edit menu
        edit_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Düzenle", menu=edit_menu)
        edit_menu.add_command(label="Orijinale Dön (Ctrl+Z)", command=self.reset_image)
        
        # View menu
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Görünüm", menu=view_menu)
        view_menu.add_command(label="Histogram Göster", command=self.show_histogram)
    
    def setup_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Processed frame
        self.processed_frame = ttk.Frame(self.processed_window, padding="10")
        self.processed_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image processing operations
        self.operations_frame = ttk.LabelFrame(self.main_frame, text="İşlemler", padding="5")
        self.operations_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # First row - Edge and filters
        ttk.Button(self.operations_frame, text="Kenar Algılama", command=self.edge_detection).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Ortalama Filtre", command=self.mean_filter).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Medyan Filtre", command=self.median_filter).grid(row=0, column=2, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Temizle", command=self.clear_image).grid(row=0, column=3, padx=5, pady=2)
        
        # Second row - Enhancement
        ttk.Button(self.operations_frame, text="Keskinleştirme", command=self.sharpen).grid(row=1, column=0, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Yumuşatma", command=self.smooth).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Döndürme", command=self.rotate).grid(row=1, column=2, padx=5, pady=2)
        
        # Third row - Histogram and threshold
        ttk.Button(self.operations_frame, text="Histogram", command=self.show_histogram).grid(row=2, column=0, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Histogram Eşitleme", command=self.histogram_equalization).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Kontrast Germe", command=self.contrast_stretch).grid(row=2, column=2, padx=5, pady=2)
        
        # Fourth row - New operations
        ttk.Button(self.operations_frame, text="Yatay Aynalama", command=lambda: self.mirror('horizontal')).grid(row=3, column=0, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Dikey Aynalama", command=lambda: self.mirror('vertical')).grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Ağırlık Merkezi", command=self.center_of_gravity).grid(row=3, column=2, padx=5, pady=2)
        
        # Fifth row - Morphological operations
        ttk.Button(self.operations_frame, text="Genişletme", command=self.dilate).grid(row=4, column=0, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Aşındırma", command=self.erode).grid(row=4, column=1, padx=5, pady=2)
        ttk.Button(self.operations_frame, text="Kapur Eşikleme", command=self.kapur_threshold).grid(row=4, column=2, padx=5, pady=2)
        
        # Thresholding frame
        self.threshold_frame = ttk.LabelFrame(self.main_frame, text="Manuel Eşikleme", padding="5")
        self.threshold_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Threshold value slider
        ttk.Label(self.threshold_frame, text="Eşik Değeri:").grid(row=0, column=0, padx=5)
        self.threshold_scale = Scale(self.threshold_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                   command=self.manual_threshold, length=200)
        self.threshold_scale.grid(row=0, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        
        # Threshold method selection
        ttk.Label(self.threshold_frame, text="Metod:").grid(row=0, column=2, padx=5)
        methods = ['İkili', 'Ters İkili', 'Kesme', 'Sıfıra', 'Ters Sıfıra']
        method_menu = ttk.OptionMenu(self.threshold_frame, self.threshold_method, 'İkili', *methods,
                                   command=lambda _: self.manual_threshold(self.threshold_scale.get()))
        method_menu.grid(row=0, column=3, padx=5)
        
        # Apply button
        ttk.Button(self.threshold_frame, text="Uygula", 
                  command=lambda: self.manual_threshold(self.threshold_scale.get())).grid(row=0, column=4, padx=5)
        
        # Image display
        self.original_label = ttk.Label(self.main_frame)
        self.original_label.grid(row=3, column=0, pady=10)
        
        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.grid(row=0, column=0, pady=10)

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.current_image = self.original_image.copy()
            self.rotation_angle = 0
            self.display_image()

    def contrast_stretch(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        
        # Convert to LAB color space
        lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl,a,b))
        
        # Convert back to BGR
        self.processed_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        self.current_image = self.processed_image.copy()
        self.display_image()

    def display_image(self):
        if self.original_image is not None:
            # Display original image
            image_pil = Image.fromarray(self.original_image)
            display_size = (400, 300)
            image_pil.thumbnail(display_size, Image.LANCZOS)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.original_label.configure(image=image_tk)
            self.original_label.image = image_tk
        
        if self.processed_image is not None:
            # Display processed image
            image_pil = Image.fromarray(self.processed_image)
            display_size = (400, 300)
            image_pil.thumbnail(display_size, Image.LANCZOS)
            image_tk = ImageTk.PhotoImage(image_pil)
            self.processed_label.configure(image=image_tk)
            self.processed_label.image = image_tk

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png *.gif *.tiff")]
        )
        if file_path:
            # Use PIL to load image and convert to numpy array
            pil_image = Image.open(file_path)
            self.original_image = np.array(pil_image)
            if len(self.original_image.shape) == 2:  # If grayscale
                self.original_image = np.stack([self.original_image] * 3, axis=-1)
            self.processed_image = self.original_image.copy()
            self.current_image = self.original_image.copy()
            self.rotation_angle = 0
            self.display_image()
    
    def apply_operation(self, operation_func):
        """Generic method to apply any operation while maintaining image state"""
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        
        # Apply the operation to the current state of the image
        self.processed_image = operation_func(self.current_image)
        # Update current state
        self.current_image = self.processed_image.copy()
        self.display_image()

    def edge_detection(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        self.processed_image = CustomImageProcessing.edge_detection(self.current_image)
        self.current_image = self.processed_image.copy()
        self.display_image()
    
    def mean_filter(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        self.processed_image = CustomImageProcessing.mean_filter(self.current_image)
        self.current_image = self.processed_image.copy()
        self.display_image()
    
    def median_filter(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        self.processed_image = CustomImageProcessing.median_filter(self.current_image)
        self.current_image = self.processed_image.copy()
        self.display_image()
    
    def sharpen(self):
        def operation(img):
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(img, -1, kernel)
        self.apply_operation(operation)
    
    def smooth(self):
        def operation(img):
            return cv2.GaussianBlur(img, (5,5), 0)
        self.apply_operation(operation)
    
    def rotate(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        
        # Increment rotation angle by 90 degrees
        self.rotation_angle = (self.rotation_angle + 90) % 360
        
        # Use custom rotation
        self.processed_image = CustomImageProcessing.rotate_image(self.original_image, self.rotation_angle)
        self.current_image = self.processed_image.copy()
        self.display_image()
    
    def show_histogram(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        
        # Create a figure with two subplots side by side
        plt.figure(figsize=(12,5))
        
        # Original image histogram
        plt.subplot(121)
        color = ('b','g','r')
        orig_hists = CustomImageProcessing.calculate_histogram(self.original_image)
        for i, (hist, col) in enumerate(zip(orig_hists, color)):
            plt.plot(hist, color=col)
        plt.title('Orijinal Görüntü Histogramı')
        plt.xlabel('Piksel Değeri')
        plt.ylabel('Piksel Sayısı')
        
        # Current image histogram
        plt.subplot(122)
        curr_hists = CustomImageProcessing.calculate_histogram(self.current_image)
        for i, (hist, col) in enumerate(zip(curr_hists, color)):
            plt.plot(hist, color=col)
        plt.title('Güncel Görüntü Histogramı')
        plt.xlabel('Piksel Değeri')
        plt.ylabel('Piksel Sayısı')
        
        plt.tight_layout()
        plt.show()
    
    def histogram_equalization(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        
        self.processed_image = CustomImageProcessing.histogram_equalization(self.current_image)
        self.current_image = self.processed_image.copy()
        self.display_image()
        
        # Show histograms
        self.show_histogram()

    def thresholding(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.current_image = self.processed_image.copy()
        self.display_image()

    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Hata", "Kaydedilecek görüntü yok!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"),
                                                          ("JPEG files", "*.jpg"),
                                                          ("All files", "*.*")])
        if file_path:
            Image.fromarray(self.processed_image).save(file_path)

    def mirror(self, direction):
        def operation(img):
            return cv2.flip(img, 1 if direction == 'horizontal' else 0)
        self.apply_operation(operation)

    def center_of_gravity(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Draw center of gravity
            img_copy = self.current_image.copy()
            cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(img_copy, (cx-20, cy), (cx+20, cy), (0, 0, 255), 2)
            cv2.line(img_copy, (cx, cy-20), (cx, cy+20), (0, 0, 255), 2)
            
            self.processed_image = img_copy
            self.current_image = self.processed_image.copy()
            self.display_image()
            
            messagebox.showinfo("Ağırlık Merkezi", f"X: {cx}, Y: {cy}")

    def dilate(self):
        def operation(img):
            kernel = np.ones((5,5), np.uint8)
            return cv2.dilate(img, kernel, iterations=1)
        self.apply_operation(operation)

    def erode(self):
        def operation(img):
            kernel = np.ones((5,5), np.uint8)
            return cv2.erode(img, kernel, iterations=1)
        self.apply_operation(operation)

    def manual_threshold(self, value):
        if self.current_image is None:
            return
            
        # Convert method names to internal values
        method_map = {
            'İkili': 'binary',
            'Ters İkili': 'binary_inv',
            'Kesme': 'trunc',
            'Sıfıra': 'tozero',
            'Ters Sıfıra': 'tozero_inv'
        }
        
        method = method_map[self.threshold_method.get()]
        
        # Apply thresholding
        self.processed_image = CustomImageProcessing.manual_threshold(
            self.current_image, 
            int(float(value)), 
            method
        )
        self.current_image = self.processed_image.copy()
        self.display_image()

    def kapur_threshold(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        
        max_entropy = float('-inf')
        threshold = 0
        
        for t in range(1, 255):
            p1 = hist[:t].sum()
            p2 = hist[t:].sum()
            
            if p1 > 0 and p2 > 0:
                h1 = -np.sum(hist[:t] * np.log2(hist[:t] + 1e-10)) / p1 if p1 > 0 else 0
                h2 = -np.sum(hist[t:] * np.log2(hist[t:] + 1e-10)) / p2 if p2 > 0 else 0
                
                entropy = h1 + h2
                if entropy > max_entropy:
                    max_entropy = entropy
                    threshold = t
        
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.current_image = self.processed_image.copy()
        self.display_image()

    def otsu_threshold(self):
        if self.current_image is None:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")
            return
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.current_image = self.processed_image.copy()
        self.display_image()

    def clear_image(self):
        """Reset the current image back to the original uploaded image"""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.current_image = self.original_image.copy()
            self.rotation_angle = 0
            self.display_image()
        else:
            messagebox.showerror("Hata", "Lütfen önce bir görüntü yükleyin!")

class CustomImageProcessing:
    @staticmethod
    def calculate_histogram(image):
        """Calculate histogram for each channel manually"""
        if len(image.shape) == 3:
            histograms = []
            for channel in range(3):
                hist = np.zeros(256)
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        hist[image[i, j, channel]] += 1
                histograms.append(hist)
            return histograms
        else:
            hist = np.zeros(256)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    hist[image[i, j]] += 1
            return [hist]

    @staticmethod
    def mean_filter(image, kernel_size=5):
        """Apply mean filter manually"""
        pad = kernel_size // 2
        result = np.zeros_like(image)
        
        # Pad the image
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        
        for i in range(pad, padded.shape[0] - pad):
            for j in range(pad, padded.shape[1] - pad):
                for c in range(3):  # For each color channel
                    window = padded[i-pad:i+pad+1, j-pad:j+pad+1, c]
                    result[i-pad, j-pad, c] = np.mean(window)
                    
        return result.astype(np.uint8)

    @staticmethod
    def median_filter(image, kernel_size=5):
        """Apply median filter manually"""
        pad = kernel_size // 2
        result = np.zeros_like(image)
        
        # Pad the image
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        
        for i in range(pad, padded.shape[0] - pad):
            for j in range(pad, padded.shape[1] - pad):
                for c in range(3):  # For each color channel
                    window = padded[i-pad:i+pad+1, j-pad:j+pad+1, c]
                    result[i-pad, j-pad, c] = np.median(window)
                    
        return result.astype(np.uint8)

    @staticmethod
    def edge_detection(image):
        """Custom edge detection using Sobel operators"""
        # Convert to grayscale first
        if len(image.shape) == 3:
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image
            
        # Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Pad the image
        padded = np.pad(gray, ((1, 1), (1, 1)), mode='edge')
        
        # Output image
        edges = np.zeros_like(gray)
        
        # Apply operators
        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                window = padded[i-1:i+2, j-1:j+2]
                gx = np.sum(window * sobel_x)
                gy = np.sum(window * sobel_y)
                edges[i-1, j-1] = min(255, np.sqrt(gx**2 + gy**2))
                
        # Convert back to 3 channels
        return np.stack([edges] * 3, axis=-1).astype(np.uint8)

    @staticmethod
    def rotate_image(image, angle):
        """Rotate image by given angle"""
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Calculate new image dimensions
        height, width = image.shape[:2]
        cos_theta = np.abs(np.cos(theta))
        sin_theta = np.abs(np.sin(theta))
        new_width = int(width * cos_theta + height * sin_theta)
        new_height = int(height * cos_theta + width * sin_theta)
        
        # Create output image
        rotated = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
        # Calculate center points
        old_center = (width // 2, height // 2)
        new_center = (new_width // 2, new_height // 2)
        
        # Rotation matrix
        for i in range(new_height):
            for j in range(new_width):
                # Translate to origin
                x = j - new_center[0]
                y = i - new_center[1]
                
                # Rotate
                old_x = int(x * np.cos(theta) - y * np.sin(theta) + old_center[0])
                old_y = int(x * np.sin(theta) + y * np.cos(theta) + old_center[1])
                
                # Check if within bounds
                if 0 <= old_x < width and 0 <= old_y < height:
                    rotated[i, j] = image[old_y, old_x]
                    
        return rotated

    @staticmethod
    def histogram_equalization(image):
        """Perform histogram equalization manually"""
        if len(image.shape) == 3:
            # Process each channel separately
            result = np.zeros_like(image)
            for c in range(3):
                channel = image[..., c]
                # Calculate histogram
                hist = np.zeros(256)
                for i in range(channel.shape[0]):
                    for j in range(channel.shape[1]):
                        hist[channel[i, j]] += 1
                        
                # Calculate cumulative distribution
                cdf = hist.cumsum()
                cdf_normalized = cdf * 255 / cdf[-1]
                
                # Apply equalization
                for i in range(channel.shape[0]):
                    for j in range(channel.shape[1]):
                        result[i, j, c] = cdf_normalized[channel[i, j]]
                        
            return result.astype(np.uint8)
        else:
            return np.stack([CustomImageProcessing.histogram_equalization(image)] * 3, axis=-1)

    @staticmethod
    def manual_threshold(image, threshold_value, method='binary'):
        """
        Apply manual thresholding with different methods
        method: 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv'
        """
        if len(image.shape) == 3:
            # Convert to grayscale using weighted sum
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image.copy()
            
        result = np.zeros_like(gray)
        
        if method == 'binary':
            result[gray > threshold_value] = 255
        elif method == 'binary_inv':
            result[gray <= threshold_value] = 255
        elif method == 'trunc':
            result = np.minimum(gray, threshold_value)
        elif method == 'tozero':
            result = np.where(gray > threshold_value, gray, 0)
        elif method == 'tozero_inv':
            result = np.where(gray <= threshold_value, gray, 0)
            
        # Convert back to 3 channels
        return np.stack([result] * 3, axis=-1).astype(np.uint8)

def main():
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main() 