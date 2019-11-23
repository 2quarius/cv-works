# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'robertsdlg.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtWidgets


class Gauss_Dialog(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Gauss_Dialog,self).__init__(parent)

        self.setObjectName("Dialog")
        self.resize(491, 149)
        self.verticalLayoutWidget = QtWidgets.QWidget(self)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(3, 19, 481, 121))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.ksize = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.ksize.setProperty("int", 0)
        self.ksize.setObjectName("ksize")
        self.horizontalLayout.addWidget(self.ksize)
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.sigma = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.sigma.setProperty("float", 1.0)
        self.sigma.setObjectName("sigma")
        self.horizontalLayout.addWidget(self.sigma)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.verticalLayoutWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi()
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        # self.sigma.editingFinished.connect(Dialog.accept)
        # self.ksize.editingFinished.connect(Dialog.accept)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Roberts Operator Arguments"))
        # Dialog.setToolTip(_translate("Dialog", "<html><head/><body><p><br/></p></body></html>"))
        # Dialog.setWhatsThis(_translate("Dialog", "<html><head/><body><p><br/></p></body></html>"))
        self.label.setText(_translate("Dialog", "Kernel Size"))
        self.label_2.setText(_translate("Dialog", "Sigma"))
    def result(self):
        return self.ksize.text(),self.sigma.text()
