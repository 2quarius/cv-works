import os
import sys
from PyQt5.QtCore import  Qt, PYQT_VERSION_STR, QT_VERSION_STR, QSettings, QTimer, QFile, QFileInfo, \
    QVariant
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QLabel, QListWidget, QApplication, QFrame, QAction, QActionGroup, \
    QMessageBox, QSpinBox, QFileDialog, QInputDialog
from PyQt5.QtGui import QImage, QKeySequence, QPixmap, QImageReader, QImageWriter, QPainter
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
import qtawesome
import gaussdlg
import mdlg
import platform
import resizedlg
import sedlg
import helpform
import filters.convolution as convolution
import filters.filters as filters
import morphology.edgeDetection as edgeDetection
import morphology.gradient as gradient
import morphology.reconstruction as reconstruction
import stack
import qimage2ndarray
import cv2
import numpy as np
import json

__version__ = "1.0.0"


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        """
        constructor
        :param parent: reference to the parent widget
        @:type QWidget
        """
        super(MainWindow, self).__init__(parent)

        self.originImage = QImage()
        self.currentImage = QImage()
        # for roll back
        # stack()
        self.imgStack = stack.Stack()
        self.cvimgstack = stack.Stack()

        self.dirty = False
        self.filename = None

        self.mirroredvertically = False
        self.mirroredhorizontally = False

        self.imageLabel = QLabel()
        self.imageLabel.setMinimumSize(200, 200)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.setCentralWidget(self.imageLabel)

        logDockWidget = QDockWidget("log", self)
        logDockWidget.setObjectName("LogDockWidget")
        logDockWidget.setAllowedAreas(Qt.LeftDockWidgetArea |
                                      Qt.RightDockWidgetArea)
        self.listWidget = QListWidget()
        logDockWidget.setWidget(self.listWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, logDockWidget)

        self.printer = None

        self.sizeLabel = QLabel()
        self.sizeLabel.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        status.addPermanentWidget(self.sizeLabel)
        status.showMessage("Ready", 5000)

        fileOpenAction = self.createAction("&Open", self.fileOpen, QKeySequence.Open, "fa.folder-open-o",
                                           "Open an existing image file")
        fileSaveAction = self.createAction("&Save", self.fileSave, QKeySequence.Save, "fa.floppy-o", "Save the image")
        fileSaveAsAction = self.createAction("Save &As...", self.fileSaveAs, icon="fa.bookmark-o",
                                             tip="Save the image using a new name")
        filePrintAction = self.createAction("&Print", self.filePrint, QKeySequence.Print, "fa.print", "Print the image")
        fileQuitAction = self.createAction("&Quit", self.close, "Ctrl+Q", "fa.times-circle", "Close the application")

        editBackAction = self.createAction("&Undo", self.editUndo, "Ctrl+B", "fa.undo", "Back to last change")

        editZoomAction = self.createAction("&Zoom...", self.editZoom,
                                           "Alt+Z", "fa.search-plus", "Zoom the image")

        editResizeAction = self.createAction("&Resize...",
                                             self.editResize, "Ctrl+R", "fa.arrows-alt",
                                             "Resize the image")
        mirrorGroup = QActionGroup(self)
        # editUnMirrorAction = self.createAction("&Unmirror",
        #                                        self.editUnMirror, "Ctrl+U", "fa.opera",
        #                                        "Unmirror the image", True, "toggled(bool)")
        # mirrorGroup.addAction(editUnMirrorAction)
        editMirrorHorizontalAction = self.createAction(
            "Mirror &Horizontally", self.editMirrorHorizontal,
            "Ctrl+H", "fa.arrows-h",
            "Horizontally mirror the image", True, "toggled(bool)")
        mirrorGroup.addAction(editMirrorHorizontalAction)
        editMirrorVerticalAction = self.createAction(
            "Mirror &Vertically", self.editMirrorVertical,
            "Ctrl+V", "fa.arrows-v",
            "Vertically mirror the image", True, "toggled(bool)")
        mirrorGroup.addAction(editMirrorVerticalAction)
        # editUnMirrorAction.setChecked(True)

        convolveGroup = QActionGroup(self)
        editRobertsConvolve = self.createAction(
            "Roberts &Operator", self.editRoberts, icon="fa.registered",
            tip="Convolve using Roberts operator", checkable=True, signal="toggled(bool)")
        convolveGroup.addAction(editRobertsConvolve)
        editPrewittConvolve = self.createAction(
            "Prewitt &Operator", self.editPrewitt, icon="fa.product-hunt",
            tip="Convolve using Prewitt operator", checkable=True, signal="toggled(bool)")
        convolveGroup.addAction(editPrewittConvolve)
        editSobelConvolve = self.createAction(
            "Sobel &Operator", self.editSobel, icon="fa.superpowers",
            tip="Convolve using Sobel operator", checkable=True, signal="toggled(bool)")
        convolveGroup.addAction(editSobelConvolve)

        filterGroup = QActionGroup(self)
        editGaussianFilter = self.createAction(
            "Gaussian &Filter", self.editGauss, icon="fa.google",
            tip="Gaussian filter", checkable=True, signal="toggled(bool)")
        filterGroup.addAction(editGaussianFilter)
        editMeanFilter = self.createAction(
            "Mean &Filter", self.editMean, icon="fa.meetup",
            tip="Mean filter", checkable=True, signal="toggled(bool)")
        filterGroup.addAction(editMeanFilter)
        editMedianFilter = self.createAction(
            "Median &Filter", self.editMedian, icon="fa.medium",
            tip="Median filter", checkable=True, signal="toggled(bool)")
        filterGroup.addAction(editMedianFilter)

        morphGroup = QActionGroup(self)
        editEdgeDetection = self.createAction(
            "Edge &Detection", self.editEdgeDetection, icon="fa.envira",
            tip="Morphological edge detection", checkable=True, signal="toggled(bool)")
        morphGroup.addAction(editEdgeDetection)
        editReconstruction = self.createAction(
            "&Reconstruction", self.editReconstruction, icon="fa.ravelry",
            tip="Morphological Reconstruction", checkable=True, signal="toggled(bool)")
        morphGroup.addAction(editReconstruction)
        editGradient = self.createAction(
            "&Gradient", self.editGradient, icon="fa.signal",
            tip="Morphological gradient", checkable=True, signal="toggled(bool)")
        morphGroup.addAction(editGradient)

        helpAboutAction = self.createAction("&About Image Changer",
                                            self.helpAbout)
        helpHelpAction = self.createAction("&Help", self.helpHelp,
                                           QKeySequence.HelpContents)

        # file Menu
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenuActions = (fileOpenAction,
                                fileSaveAction, fileSaveAsAction, None, filePrintAction,
                                fileQuitAction)

        self.fileMenu.aboutToShow.connect(self.updateFileMenu)
        # edit Menu
        editMenu = self.menuBar().addMenu("&Edit")
        self.addActions(editMenu, (editZoomAction, editResizeAction, editBackAction))
        # mirrorMenu
        mirrorMenu = editMenu.addMenu(qtawesome.icon("fa.modx"),
                                      "&Mirror")
        self.addActions(mirrorMenu, (editMirrorHorizontalAction, editMirrorVerticalAction))
        # convolveMenu
        convolveMenu = editMenu.addMenu(qtawesome.icon("fa.connectdevelop"), "&Convolve")
        self.addActions(convolveMenu, (editRobertsConvolve, editPrewittConvolve, editSobelConvolve))

        # filterMenu
        filterMenu = editMenu.addMenu(qtawesome.icon("fa.filter"), "&Filter")
        self.addActions(filterMenu, (editGaussianFilter, editMeanFilter, editMedianFilter))

        # morphology
        morphoMenu = editMenu.addMenu(qtawesome.icon("fa.mixcloud"), "&Morphology")
        self.addActions(morphoMenu,(editEdgeDetection,editReconstruction,editGradient))

        # help Menu
        helpMenu = self.menuBar().addMenu("&Help")
        self.addActions(helpMenu, (helpAboutAction, helpHelpAction))

        # tool bar
        # fileToolbar = self.addToolBar("File")
        # fileToolbar.setObjectName("FileToolBar")
        # self.addActions(fileToolbar, (fileNewAction, fileOpenAction,
        #                               fileSaveAsAction))
        editToolbar = self.addToolBar("Edit")
        editToolbar.setObjectName("EditToolBar")
        self.addActions(editToolbar, (editBackAction, editMirrorVerticalAction, editMirrorHorizontalAction,
                                      editRobertsConvolve, editPrewittConvolve, editSobelConvolve,
                                      editGaussianFilter, editMeanFilter, editMedianFilter,
                                      editEdgeDetection,editReconstruction,editGradient))
        self.zoomSpinBox = QSpinBox()
        self.zoomSpinBox.setRange(1, 400)
        self.zoomSpinBox.setSuffix(" %")
        self.zoomSpinBox.setValue(100)
        self.zoomSpinBox.setToolTip("Zoom the image")
        self.zoomSpinBox.setStatusTip(self.zoomSpinBox.toolTip())
        self.zoomSpinBox.setFocusPolicy(Qt.NoFocus)
        self.zoomSpinBox.valueChanged[int].connect(self.showImage)
        editToolbar.addWidget(self.zoomSpinBox)

        self.addActions(self.imageLabel, (editBackAction, editMirrorVerticalAction, editMirrorHorizontalAction,
                                          editRobertsConvolve, editPrewittConvolve, editSobelConvolve,
                                          editGaussianFilter, editMeanFilter, editMedianFilter,
                                          editEdgeDetection,editReconstruction,editGradient))

        self.resetableActions = ((editRobertsConvolve, True),
                                 (editPrewittConvolve, True),
                                 (editSobelConvolve, True),
                                 (editGaussianFilter, True),
                                 (editMeanFilter, True),
                                 (editMedianFilter, True))

        settings = QSettings("MyCompany", "MyApp")
        self.recentFiles = []
        if settings.value("RecentFiles"):
            self.recentFiles = settings.value("RecentFiles")
        # self.recentFiles=settings.value("RecentFiles")
        #
        # self.restoreGeometry(QByteArray(settings.value("MainWindow/Geometry")))
        # self.restoreState(QByteArray(settings.value("MainWindow/State")))
        self.setWindowTitle("Image Changer")
        self.updateFileMenu()
        QTimer.singleShot(0, self.loadInitialFile)

    def createAction(self, text, slot=None, shortcut=None, icon=None,
                     tip=None, checkable=False, signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(qtawesome.icon(icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None and signal == "triggered()":
            action.triggered.connect(slot)
        if slot is not None and signal == "toggled(bool)":
            action.toggled[bool].connect(slot)
        if checkable:
            action.setCheckable(True)
        return action

    def addActions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def updateFileMenu(self):
        self.fileMenu.clear()
        self.addActions(self.fileMenu, self.fileMenuActions[:-1])
        current = self.filename
        recentFiles = []

        for fname in self.recentFiles:
            if fname != current and QFile.exists(fname):
                recentFiles.append(fname)
        if recentFiles:
            self.fileMenu.addSeparator()
            for i, fname in enumerate(recentFiles):
                action = QAction(qtawesome.icon("fa.folder-open"), "&{0}{1}".format(i + 1, QFileInfo(fname).fileName()),
                                 self)
                action.setData(QVariant(fname))
                action.triggered[bool].connect(self.loadFile)
                self.fileMenu.addAction(action)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.fileMenuActions[-1])

    def okToContinue(self):
        if self.dirty:
            reply = QMessageBox.question(self,
                                         "Image Changer - Unsaved Changes",
                                         "Save unsaved changes?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return False
            elif reply == QMessageBox.Yes:
                return self.fileSave()
        return True

    def fileOpen(self):
        if not self.okToContinue():
            return
        dir = (os.path.dirname(self.filename)
               if self.filename is not None else ".")
        formats = (["*.{0}".format(format.data().decode("ascii").lower())
                    for format in QImageReader.supportedImageFormats()])
        fname, tpye = QFileDialog.getOpenFileName(self,
                                                  "Image Changer - Choose Image", dir,
                                                  "Image files ({0})".format(" ".join(formats)))
        if fname:
            self.loadFile(True, fname)

    def fileSave(self):
        if self.originImage.isNull():
            return True
        if self.filename is None:
            return self.fileSaveAs()
        else:
            if self.currentImage.save(self.filename, None):
                self.updateStatus("Saved as {0}".format(self.filename))
                self.dirty = False
                return True
            else:
                self.updateStatus("Failed to save {0}".format(
                    self.filename))
                return False

    def fileSaveAs(self):
        if self.originImage.isNull():
            return True
        fname = self.filename if self.filename is not None else "."
        formats = (["*.{0}".format(format.data().decode("ascii").lower())
                    for format in QImageWriter.supportedImageFormats()])
        fname, tpye = QFileDialog.getSaveFileName(self,
                                                  "Image Changer - Save Image", fname,
                                                  "Image files ({0})".format(" ".join(formats)))
        if fname:
            if "." not in fname:
                fname += ".png"
            self.addRecentFile(fname)
            self.filename = fname
            return self.fileSave()
        return False

    def filePrint(self):
        if self.currentImage.isNull():
            return
        if self.printer is None:
            self.printer = QPrinter(QPrinter.HighResolution)
            self.printer.setPageSize(QPrinter.Letter)
        form = QPrintDialog(self.printer, self)
        if form.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.currentImage.size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(),
                                size.height())
            painter.drawImage(0, 0, self.currentImage)

    def loadFile(self, actiontrigger=False, fname=None):
        if fname is None:
            action = self.sender()
            if isinstance(action, QAction):
                fname = str(action.data())
                if not self.okToContinue():
                    return
            else:
                return
        print(fname)
        if fname:
            self.filename = None
            image = QImage(fname)
            cvimg = cv2.imread(fname)
            if image.isNull():
                message = "Failed to read {0}".format(fname)
            else:
                self.addRecentFile(fname)
                self.originImage = QImage()
                for action, check in self.resetableActions:
                    action.setChecked(check)
                self.originImage = image
                self.currentImage = self.originImage
                self.imgStack.push(self.currentImage)
                self.cvimg = cvimg
                self.cvimgstack.push(self.cvimg)
                self.filename = fname
                self.showImage()
                self.dirty = False
                self.sizeLabel.setText("{0} x {1}".format(
                    image.width(), image.height()))
                message = "Loaded {0}".format(os.path.basename(fname))
            self.updateStatus(message)

    def loadInitialFile(self):
        settings = QSettings()
        fname = str(settings.value("LastFile"))
        if fname and QFile.exists(fname):
            self.loadFile(fname)

    def addRecentFile(self, fname):
        if fname is None:
            return
        if fname not in self.recentFiles:
            self.recentFiles.insert(0, fname)
            while len(self.recentFiles) > 9:
                self.recentFiles.pop()

    def updateStatus(self, message):
        self.statusBar().showMessage(message, 5000)
        self.listWidget.addItem(message)
        if self.filename is not None:
            self.setWindowTitle("Image Changer - {0}[*]".format(
                os.path.basename(self.filename)))
        elif not self.originImage.isNull():
            self.setWindowTitle("Image Changer - Unnamed[*]")
        else:
            self.setWindowTitle("Image Changer[*]")
        self.setWindowModified(self.dirty)

    def editUndo(self):
        # TODO: stack pop
        bottom = self.imgStack.pop()
        cv = self.cvimgstack.pop()
        if not self.imgStack.isEmpty():
            self.currentImage = self.imgStack.peek()
            self.cvimg = self.cvimgstack.peek()
            self.updateStatus("Undo last change")
        else:
            self.imgStack.push(bottom)
            self.cvimgstack.push(cv)
            self.dirty = False
            self.updateStatus("Already the oldest change can roll back")
        self.showImage()

    def editZoom(self):
        if self.currentImage.isNull():
            return
        percent, ok = QInputDialog.getInt(self,
                                          "Image Changer - Zoom", "Percent:",
                                          self.zoomSpinBox.value(), 1, 400)
        if ok:
            self.zoomSpinBox.setValue(percent)

    # def editUnMirror(self, on):
    #     if self.currentImage.isNull():
    #         return
    #     if self.mirroredhorizontally:
    #         self.editMirrorHorizontal(False)
    #     if self.mirroredvertically:
    #         self.editMirrorVertical(False)

    def editMirrorHorizontal(self, on):
        if self.currentImage.isNull() or not on:
            return
        self.currentImage = self.currentImage.mirrored(True, False)
        self.imgStack.push(self.currentImage)
        self.cvimg = cv2.flip(self.cvimg,1)
        self.cvimgstack.push(self.cvimg)
        self.showImage()
        self.mirroredhorizontally = not self.mirroredhorizontally
        self.dirty = True
        self.updateStatus(("Mirrored Horizontally"
                           if on else "Unmirrored Horizontally"))

    def editMirrorVertical(self, on):
        if self.currentImage.isNull() or not on:
            return
        self.currentImage = self.currentImage.mirrored(False, True)
        self.imgStack.push(self.currentImage)
        self.cvimg = cv2.flip(self.cvimg,0)
        self.cvimgstack.push(self.cvimg)
        self.showImage()
        self.mirroredvertically = not self.mirroredvertically
        self.dirty = True
        self.updateStatus(("Mirrored Vertically"
                           if on else "Unmirrored Vertically"))

    def editResize(self):
        if self.currentImage.isNull():
            return
        form = resizedlg.ResizeDlg(self.currentImage.width(),
                                   self.currentImage.height(), self)
        if form.exec_():
            width, height = form.result()
            if (width == self.currentImage.width() and
                    height == self.currentImage.height()):
                self.statusBar().showMessage("Resized to the same size",
                                             5000)
            else:
                self.currentImage = self.imgStack.push(self.currentImage.scaled(width, height))
                self.cvimg = self.cvimgstack.push(cv2.resize(self.cvimg,(width,height),interpolation=cv2.INTER_CUBIC))
                self.showImage()
                self.dirty = True
                size = "{0} x {1}".format(self.currentImage.width(),
                                          self.currentImage.height())
                self.sizeLabel.setText(size)
                self.updateStatus("Resized to {0}".format(size))

    def editRoberts(self, on):
        if self.currentImage.isNull() or not on:
            return
        self.cvimg = convolution.roberts_convolve(self.cvimg)
        self.currentImage = qimage2ndarray.array2qimage(self.cvimg)
        self.imgStack.push(self.currentImage)
        self.cvimgstack.push(self.cvimg)
        self.showImage()
        self.dirty = True
        self.updateStatus("Convolved(Roberts)"
                          if on else "Unconvolved(Roberts)")

    def editPrewitt(self, on):
        if self.currentImage.isNull() or not on:
            return
        self.cvimg = convolution.prewitt_convolve(self.cvimg)
        self.currentImage = qimage2ndarray.array2qimage(self.cvimg)
        self.imgStack.push(self.currentImage)
        self.cvimgstack.push(self.cvimg)
        self.showImage()
        self.dirty = True
        self.updateStatus("Convolved(Prewitt)"
                          if on else "Unconvolved(Prewitt)")

    def editSobel(self, on):
        if self.currentImage.isNull() or not on:
            return
        self.cvimg = convolution.sobel_convolve(self.cvimg)
        self.currentImage = qimage2ndarray.array2qimage(self.cvimg)
        self.imgStack.push(self.currentImage)
        self.cvimgstack.push(self.cvimg)
        self.showImage()
        self.dirty = True
        self.updateStatus("Convolved(Sobel)"
                          if on else "Unconvolved(Sobel)")

    def editGauss(self, on):
        if self.currentImage.isNull() or not on:
            return
        form = gaussdlg.Gauss_Dialog(self)
        if form.exec_():
            k, s = form.result()
            if k and s:
                self.cvimg = filters.GaussBlur(self.cvimg, int(k), float(s)).gauss()
                self.currentImage = qimage2ndarray.array2qimage(self.cvimg)
                self.imgStack.push(self.currentImage)
                self.cvimgstack.push(self.cvimg)
                self.updateStatus("Filtered(Gaussian)"
                                  if on else "Unfiltered(Gaussian)")
            else:
                self.updateStatus("wrong arguments passed")
        self.showImage()
        self.dirty = True

    def editMean(self, on):
        if self.currentImage.isNull() or not on:
            return
        form = mdlg.M_Dialog(self)
        if form.exec_():
            k = form.result()
            if k:
                self.cvimg = filters.MeanBlur(self.cvimg, int(k)).mean()
                self.currentImage = qimage2ndarray.array2qimage(self.cvimg)
                self.imgStack.push(self.currentImage)
                self.cvimgstack.push(self.cvimg)
                self.updateStatus("Filtered(Mean)"
                                  if on else "Unfiltered(Mean)")
            else:
                self.updateStatus("wrong arguments passed")
        self.showImage()
        self.dirty = True

    def editMedian(self, on):
        if self.currentImage.isNull() or not on:
            return
        form = mdlg.M_Dialog(self)
        if form.exec_():
            k = form.result()
            if k:
                self.cvimg = filters.MedianBlur(self.cvimg, int(k)).median()
                self.currentImage = qimage2ndarray.array2qimage(self.cvimg)
                self.imgStack.push(self.currentImage)
                self.cvimgstack.push(self.cvimg)
                self.updateStatus("Filtered(Median)"
                                  if on else "Unfiltered(Median)")
            else:
                self.updateStatus("wrong arguments passed")
        self.showImage()
        self.dirty = True

    def editEdgeDetection(self,on):
        if self.currentImage.isNull() or not on:
            return
        form = sedlg.SE_Dialog(self)
        if form.exec_():
            k = form.result()
            if k:
                tmp = np.array(json.loads(k))
                if type(tmp)==type(tmp[0]):
                    # do edge detection
                    self.cvimg = edgeDetection.edgeDetectionStd(self.cvimg,tmp)
                    self.currentImage = qimage2ndarray.array2qimage(self.cvimg)
                    self.imgStack.push(self.currentImage)
                    self.cvimgstack.push(self.cvimg)
                    self.updateStatus("Morphological edge detection" if on else "not morphological edge detection")
                else:
                    self.updateStatus("Wrong structure element")
            else:
                self.updateStatus("no argument")
        self.showImage()
        self.dirty = True

    def editReconstruction(self, on):
        if self.currentImage.isNull() or not on:
            return
        form = sedlg.SE_Dialog(self)
        if form.exec_():
            k = form.result()
            if k:
                tmp = np.array(json.loads(k))
                if type(tmp)==type(tmp[0]):
                    # do edge detection
                    self.cvimg = reconstruction.reconstruction(self.cvimg,tmp)
                    self.currentImage = qimage2ndarray.array2qimage(self.cvimg)
                    self.imgStack.push(self.currentImage)
                    self.cvimgstack.push(self.cvimg)
                    self.updateStatus("Morphological reconstruction" if on else "not morphological reconstruction")
                else:
                    self.updateStatus("Wrong structure element")
            else:
                self.updateStatus("no argument")
        self.showImage()
        self.dirty = True

    def editGradient(self, on):
        if self.currentImage.isNull() or not on:
            return
        form = sedlg.SE_Dialog(self)
        if form.exec_():
            k = form.result()
            if k:
                tmp = np.array(json.loads(k))
                if type(tmp)==type(tmp[0]):
                    # do edge detection
                    self.cvimg = gradient.gradient(self.cvimg,tmp)
                    self.currentImage = qimage2ndarray.array2qimage(self.cvimg)
                    self.imgStack.push(self.currentImage)
                    self.cvimgstack.push(self.cvimg)
                    self.updateStatus("Morphological gradient" if on else "not morphological gradient")
                else:
                    self.updateStatus("Wrong structure element")
            else:
                self.updateStatus("no argument")
        self.showImage()
        self.dirty = True

    def helpAbout(self):
        QMessageBox.about(self, "About Image Changer",
                          """<b>Image Changer</b> v {0}
                          <p>Copyright &copy; 2008-9 Qtrac Ltd. 
                          All rights reserved.
                          <p>This application can be used to perform
                          simple image manipulations.
                          <p>Python {1} - Qt {2} - PyQt {3} on {4}""".format(
                              __version__, platform.python_version(),
                              QT_VERSION_STR, PYQT_VERSION_STR,
                              platform.system()))

    def helpHelp(self):
        form = helpform.HelpForm("index.html", self)
        form.show()

    def showImage(self, percent=None):
        if self.currentImage.isNull():
            return
        if percent is None:
            percent = self.zoomSpinBox.value()
        factor = percent / 100.0
        width = self.currentImage.width() * factor
        height = self.currentImage.height() * factor
        image = self.currentImage.scaled(width, height, Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        if self.okToContinue():
            settings = QSettings("MyCompany", "MyApp")
            settings.setValue("LastFile", self.filename)
            settings.setValue("RecentFiles", self.recentFiles)
            settings.setValue("MainWindow/Geometry", self.saveGeometry())
            settings.setValue("MainWindow/State", self.saveState())

        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()
