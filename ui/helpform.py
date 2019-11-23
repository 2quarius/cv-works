from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtWidgets import QAction, QApplication, QDialog, QLabel, QTextBrowser, QToolBar, QVBoxLayout
from PyQt5.QtGui import QKeySequence
import qtawesome
import PyQt5.QtGui

class HelpForm(QDialog):

    def __init__(self, page, parent=None):
        super(HelpForm, self).__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAttribute(Qt.WA_GroupLeader)

        backAction = QAction(qtawesome.icon("fa.backward"), "&Back", self)
        backAction.setShortcut(QKeySequence.Back)
        homeAction = QAction(qtawesome.icon("fa.home"), "&Home", self)
        homeAction.setShortcut("Home")
        self.pageLabel = QLabel()

        toolBar = QToolBar()
        toolBar.addAction(backAction)
        toolBar.addAction(homeAction)
        toolBar.addWidget(self.pageLabel)
        self.textBrowser = QTextBrowser()

        layout = QVBoxLayout()
        layout.addWidget(toolBar)
        layout.addWidget(self.textBrowser, 1)
        self.setLayout(layout)

        backAction.triggered.connect(self.tbackward)
        homeAction.triggered.connect(self.thome)
        self.textBrowser.sourceChanged.connect(self.updatePageTitle)

        self.textBrowser.setSearchPaths([":/help"])
        self.textBrowser.setSource(QUrl(page))
        self.resize(400, 600)
        self.setWindowTitle("{0} Help".format(
            QApplication.applicationName()))

    def updatePageTitle(self):
        self.pageLabel.setText(self.textBrowser.documentTitle())

    def tbackward(self):
        self.textBrowser.backward()

    def thome(self):
        self.textBrowser.home()
