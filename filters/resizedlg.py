from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import ( QDialog, QDialogButtonBox,
                             QGridLayout, QLabel, QSpinBox)


class ResizeDlg(QDialog):

    def __init__(self, width, height, parent=None):
        super(ResizeDlg, self).__init__(parent)

        widthLabel = QLabel("&Width:")
        self.widthSpinBox = QSpinBox()
        widthLabel.setBuddy(self.widthSpinBox)
        self.widthSpinBox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.widthSpinBox.setRange(4, width * 4)
        self.widthSpinBox.setValue(width)
        heightLabel = QLabel("&Height:")
        self.heightSpinBox = QSpinBox()
        heightLabel.setBuddy(self.heightSpinBox)
        self.heightSpinBox.setAlignment(Qt.AlignRight |
                                        Qt.AlignVCenter)
        self.heightSpinBox.setRange(4, height * 4)
        self.heightSpinBox.setValue(height)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok |
                                     QDialogButtonBox.Cancel)

        layout = QGridLayout()
        layout.addWidget(widthLabel, 0, 0)
        layout.addWidget(self.widthSpinBox, 0, 1)
        layout.addWidget(heightLabel, 1, 0)
        layout.addWidget(self.heightSpinBox, 1, 1)
        layout.addWidget(buttonBox, 2, 0, 1, 2)
        self.setLayout(layout)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        self.setWindowTitle("Image Changer - Resize")

    def result(self):
        return self.widthSpinBox.value(), self.heightSpinBox.value()
