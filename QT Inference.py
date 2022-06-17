from PySide6.QtWidgets import *
from PySide6.QtUiTools import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        self.ui = loader.load("main.ui", None)
        self.ui.show()
        self.ui.setWindowTitle("Pix2Pix")
        
        self.ui.upload.clicked.connect(self.Upload_img)
        self.ui.result.clicked.connect(self.Creat_Pix2Pix_img)

    def Upload_img(self):
        dialog = QFileDialog()
        self.fname = dialog.getOpenFileName(self, 'Open file', "","Image files (*.jpg *.gif *.jpeg*.png)")
        
        self.imagepath = self.fname[0]
        self.ui.label.setStyleSheet(f"image : url({self.imagepath});")
    
    def Creat_Pix2Pix_img(self):
        try:
            self.model = load_model("facades.h5", compile=False)
            width = height = 256
            img = cv2.imread(str(self.imagepath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = cv2.resize(img, (width, height)).astype(np.float32)
            img = img.reshape(1, width, height, 3)

            generate = self.model(img,training=True)
            generate = np.squeeze(generate, axis=0)
            generate = np.array((generate +1) *127.5).astype(np.uint8)
    
            img = QImage(generate, generate.shape[1], generate.shape[0], QImage.Format_RGB888)
            img = QPixmap(img)
            self.ui.label_2.setPixmap(img)

        except:
            msg = QMessageBox()
            msg.setText("Please upload your photo first")
            msg.exec()

app = QApplication()
window = MainWindow()
app.exec()