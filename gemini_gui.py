import sys
import os
import json
import cv2 as cv
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGridLayout, QPushButton, QLabel, QFileDialog,
                               QLineEdit, QRadioButton, QButtonGroup, QStackedWidget)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

from google import genai
from google.genai import types
from gemini_test import query_by_text, query_by_image

class ImagePreviewFileDialog(QFileDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(500, 500)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid gray;")
        
        layout = self.layout()
        layout.addWidget(self.preview_label, 0, layout.columnCount(), layout.rowCount(), 1)
        
        self.currentChanged.connect(self.on_current_changed)

    def on_current_changed(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        else:
            self.preview_label.clear()
            self.preview_label.setText("No preview")

class GeminiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt")
        self.setGeometry(100, 100, 1000, 700)
        
        self.client = genai.Client()
        self.config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        self.input_image_path = None
        self.prompt_image_path = None
        self.current_display_image = None
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)
        
        # Single row layout for ultra-compact controls
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 10)
        
        # Input Element
        self.btn_select_img = QPushButton("Select Input Image")
        self.btn_select_img.clicked.connect(self.select_input_image)
        self.lbl_input_path = QLabel("None")
        
        # Prompt Type Element
        self.radio_text = QRadioButton("Text Prompt")
        self.radio_image = QRadioButton("Image Prompt")
        self.radio_text.setChecked(True)
        
        self.prompt_group = QButtonGroup()
        self.prompt_group.addButton(self.radio_text)
        self.prompt_group.addButton(self.radio_image)
        
        # Prompt Input Stack
        self.prompt_stack = QStackedWidget()
        self.prompt_stack.setMaximumHeight(30)
        
        # Text input widget
        text_widget = QWidget()
        text_layout = QHBoxLayout(text_widget)
        text_layout.setContentsMargins(0, 0, 0, 0)
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setPlaceholderText("Enter object to detect...")
        text_layout.addWidget(self.txt_prompt)
        self.prompt_stack.addWidget(text_widget)
        
        # Image prompt widget
        img_prompt_widget = QWidget()
        img_prompt_layout = QHBoxLayout(img_prompt_widget)
        img_prompt_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_select_prompt_img = QPushButton("Select Prompt Image")
        self.btn_select_prompt_img.clicked.connect(self.select_prompt_image)
        self.lbl_prompt_path = QLabel("None")
        img_prompt_layout.addWidget(self.btn_select_prompt_img)
        img_prompt_layout.addWidget(self.lbl_prompt_path)
        self.prompt_stack.addWidget(img_prompt_widget)
        
        # Run Button
        self.btn_query = QPushButton("Run Query")
        self.btn_query.clicked.connect(self.run_query)
        self.btn_query.setEnabled(False)
        
        # Add all to single horizontal row
        control_layout.addWidget(self.btn_select_img)
        control_layout.addWidget(self.lbl_input_path)
        control_layout.addSpacing(15)
        control_layout.addWidget(self.radio_text)
        control_layout.addWidget(self.radio_image)
        control_layout.addWidget(self.prompt_stack, 1)  # Stretch factor 1
        control_layout.addWidget(self.btn_query)
        
        layout.addLayout(control_layout)
        
        self.radio_text.toggled.connect(self.switch_prompt_type)
        self.radio_image.toggled.connect(self.switch_prompt_type)
        
        # Image display
        self.lbl_image_display = QLabel("Select an image to start")
        self.lbl_image_display.setAlignment(Qt.AlignCenter)
        self.lbl_image_display.setMinimumSize(400, 300)
        self.lbl_image_display.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        layout.addWidget(self.lbl_image_display, 1)

    def switch_prompt_type(self):
        if self.radio_text.isChecked():
            self.prompt_stack.setCurrentIndex(0)
        else:
            self.prompt_stack.setCurrentIndex(1)
            
    def select_input_image(self):
        dialog = ImagePreviewFileDialog(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if dialog.exec():
            selected_files = dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                self.input_image_path = file_path
                self.lbl_input_path.setText(os.path.basename(file_path))
                self.current_display_image = self.input_image_path
                self.display_image(self.input_image_path)
                self.btn_query.setEnabled(True)
            
    def select_prompt_image(self):
        dialog = ImagePreviewFileDialog(self, "Select Prompt Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if dialog.exec():
            selected_files = dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                self.prompt_image_path = file_path
                self.lbl_prompt_path.setText(os.path.basename(file_path))
            
    def display_image(self, path_or_cv_img):
        if isinstance(path_or_cv_img, str):
            pixmap = QPixmap(path_or_cv_img)
        else:
            # Assume cv2 image (BGR)
            height, width, channel = path_or_cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(path_or_cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            
        scaled_pixmap = pixmap.scaled(self.lbl_image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_image_display.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        if self.current_display_image is not None:
            self.display_image(self.current_display_image)
        super().resizeEvent(event)
        
    def run_query(self):
        if not self.input_image_path:
            return
            
        self.btn_query.setEnabled(False)
        self.lbl_image_display.setText("Processing request...")
        QApplication.processEvents()
        
        try:
            img = Image.open(self.input_image_path)
            width, height = img.size
            
            if self.radio_text.isChecked():
                prompt = self.txt_prompt.text()
                if not prompt:
                    self.lbl_image_display.setText("Error: Prompt is empty.")
                    self.btn_query.setEnabled(True)
                    return
                response = query_by_text(self.client, self.config, img, prompt)
            else:
                if not self.prompt_image_path:
                    self.lbl_image_display.setText("Error: Prompt image not selected.")
                    self.btn_query.setEnabled(True)
                    return
                img2 = Image.open(self.prompt_image_path)
                response = query_by_image(self.client, self.config, img, img2)
                
            bounding_boxes = json.loads(response.text)
            
            cv_img = cv.imread(self.input_image_path)
            for bounding_box in bounding_boxes:
                # Handle possible key existence to avoid index errors, wait, the text prompt requires a different JSON key maybe?
                # Actually, the original script does bounding_box["box_2d"]
                if "box_2d" in bounding_box:
                    abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
                    abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
                    abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
                    abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
                    cv.rectangle(cv_img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                
            base_name = os.path.basename(self.input_image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join("data/output", f"{name}_bboxes{ext}")
            os.makedirs("data/output", exist_ok=True)
            cv.imwrite(output_path, cv_img)
            
            self.current_display_image = cv_img
            self.display_image(cv_img)
            
        except Exception as e:
            self.lbl_image_display.setText(f"Error: {str(e)}")
            
        self.btn_query.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GeminiApp()
    window.show()
    sys.exit(app.exec())
