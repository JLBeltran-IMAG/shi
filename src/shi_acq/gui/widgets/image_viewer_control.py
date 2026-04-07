import numpy as np
import sys

import logging
from pathlib import Path
import argparse
import acq_directories

from PySide6.QtWidgets import (
    QApplication, QWidget, QGraphicsScene, QGraphicsView, QLabel, QPushButton, QSpinBox, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMainWindow, QMessageBox, QFrame, QGraphicsPixmapItem
)
from PySide6.QtGui import QPixmap, QImage, QPixmap, QPainter
from PySide6.QtCore import Qt


def my_parser():
    parser = argparse.ArgumentParser(
        prog="ACQ",
        description="%(prog)s: This software is an automated implementation for taking_nimgs images with order",
    )

    # Defining arguments for various functionalities
    parser.add_argument("-n", "--name", required=True, type=str, help="Name of the directory.")
    # parser.add_argument("-d", "--delete", type=str, help="Delete the directory with the name specify by -d or --delete")

    return parser


class AcqViewerApp(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        # Image viewer
        self.image_scene = QGraphicsScene()
        self.image_view = QGraphicsView(self.image_scene)
        self.image_view.setRenderHints(
            self.image_view.renderHints() |
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.image_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        layout.addWidget(self.image_view)
        self.load_image("/home/beltran/absorption1.tif")


    def load_image(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.image_scene.clear()
            item = QGraphicsPixmapItem(pixmap)
            self.image_scene.addItem(item)
            self.image_scene.setSceneRect(pixmap.rect())
            self.image_view.fitInView(self.image_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)










if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = AcqViewerApp()
    viewer.show()
    sys.exit(app.exec())



