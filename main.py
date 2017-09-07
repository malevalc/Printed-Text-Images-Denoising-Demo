import os
import sys
import csv
import cv2
import numpy as np
import globals
import data
from keras.models import load_model
from qtpy import QtCore, QtGui, QtWidgets, uic
from PIL import Image
import pytesseract

qtCreatorFile = "demo.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    model_name = ''
    image = ''
    model_file = ''
    input_image = []
    output_image = []

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.ChooseInput.setEnabled(False)
        self.CleanImage.setEnabled(False)
        self.ApplyOcr.setEnabled(False)
        self.modelButton.clicked.connect(self.load_model)
        self.ChooseInput.clicked.connect(self.load_image)
        self.CleanImage.clicked.connect(self.clean_image)
        self.ApplyOcr.clicked.connect(self.perform_ocr)

    def load_model(self):
        global model_file
        self.ImagePic.setText('Original')
        self.CleanPic.setText('Cleaned')
        self.OrigOCR.setText('ORIGINAL OCR')
        self.CleanOCR.setText('CLEANED OCR')
        model_filename = QtWidgets.QFileDialog.getOpenFileName(self, "Open", os.getcwd()+"/models", "Models (*.h5)")
        model_file = os.path.basename(model_filename[0])
        model_name = model_file.replace(".h5", "")
        model_summary = os.path.join(os.getcwd(), 'models', model_name, 'summary.png')
        model_parameters = os.path.join(os.getcwd(), 'models', model_name, 'parameters.csv')
        self.nameLabel.setText("Model: " + model_name)
        self.SummaryPlot.setPixmap(QtGui.QPixmap(model_summary).scaled(480, 360))
        parameters = []
        with open(model_parameters, "r") as fileInput:
            for rows in csv.reader(fileInput):
                parameters.append(rows)
        for i, row in enumerate(parameters):
            for j, col in enumerate(row):
                item = QtWidgets.QTableWidgetItem(col)
                self.ParametersTable.setItem(i, j, item)
        self.ParametersTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.ParametersTable.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.ChooseInput.setEnabled(True)
        self.CleanImage.setEnabled(False)
        self.ApplyOcr.setEnabled(False)

    def load_image(self):
        global image
        self.OrigOCR.setText('ORIGINAL OCR')
        self.CleanOCR.setText('CLEANED OCR')
        self.CleanPic.setText('Cleaned')
        filename = QtWidgets.QFileDialog.getOpenFileName(self, "Open", os.getcwd() + "/_data", "Images (*.png)")
        image = (filename[0])
        self.ImagePic.setPixmap(QtGui.QPixmap(image).scaled(512, 256))
        self.CleanImage.setEnabled(True)
        self.ApplyOcr.setEnabled(False)

    def clean_image(self):
        global input_image
        global output_image
        self.OrigOCR.setText('ORIGINAL OCR')
        self.CleanOCR.setText('CLEANED OCR')
        model = load_model(os.path.join('models', model_file))
        input_image = data.size_handle(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
        input_image[:] = [img - 0.5 for img in input_image]
        file_name = input_image.reshape(1, globals.INPUT_ROWS, globals.INPUT_COLS, 1)
        prediction = model.predict(file_name)
        prediction = prediction.reshape(globals.INPUT_ROWS, globals.INPUT_COLS)
        output_image = np.asarray(prediction * 255.0, dtype=np.uint8)
        predicted_image = QtGui.QImage(output_image.data, output_image.shape[1], output_image.shape[0],
                                       output_image.strides[0], QtGui.QImage.Format_Indexed8)
        predicted_image.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])
        self.CleanPic.setPixmap(QtGui.QPixmap(predicted_image).scaled(512, 256))
        self.ApplyOcr.setEnabled(True)

    def perform_ocr(self):
        original_text = pytesseract.image_to_string(Image.open(image))
        clean_text = pytesseract.image_to_string(Image.fromarray(output_image, 'L'))
        original_text = original_text.replace('\n', ' ').replace('\r', '')
        clean_text = clean_text.replace('\n', ' ').replace('\r', '')
        self.OrigOCR.setText(original_text)
        self.CleanOCR.setText(clean_text)
        self.OrigOCR.setWordWrap(True)
        self.CleanOCR.setWordWrap(True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
