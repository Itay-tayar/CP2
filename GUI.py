# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from ex2 import *
import sys


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.all_frames = Frames()
        self.timer_id = -1
        self.view_point_im = None
        self.refocus_im = None
        self.left_right_prev_val = None
        self.forward_back_prev_val = None
        self.angle_prev_val = None
        self.fixed_frames_view_point = None
        self.dx_motions_vp = None
        self.timer_flag = False

    def setup_ui(self, MainWindow):
        self.mainwindow = MainWindow
        self.mainwindow.setObjectName("MainWindow")
        self.mainwindow.resize(250, 550)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(-1, -1, 250, 490))
        self.tabWidget.setObjectName("tabWidget")
        self.view_point_tab = QtWidgets.QWidget()
        self.view_point_tab.setObjectName("view_point_tab")

        self.view_point_view = QtWidgets.QLabel(self.view_point_tab)
        self.view_point_view.setGeometry(QtCore.QRect(250, 10, 0, 0))
        self.view_point_view.setText("")
        self.view_point_view.setScaledContents(True)
        self.view_point_view.setObjectName("view_point_view")

        self.view_point_frame = QtWidgets.QFrame(self.view_point_tab)
        self.view_point_frame.setGeometry(QtCore.QRect(10, 10, 230, 400))
        self.view_point_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.view_point_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.view_point_frame.setObjectName("frame")

        self.view_point_lbl = QtWidgets.QLabel(self.view_point_frame)
        self.view_point_lbl.setGeometry(QtCore.QRect(30, 10, 200, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.view_point_lbl.setFont(font)
        self.view_point_lbl.setObjectName("view_point_lbl")

        self.nums_of_frames_lbl = QtWidgets.QLabel(self.view_point_frame)
        self.nums_of_frames_lbl.setGeometry(QtCore.QRect(20, 65, 250, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.nums_of_frames_lbl.setFont(font)
        self.nums_of_frames_lbl.setObjectName("frameamountlbl")

        self.size_of_frame_lbl = QtWidgets.QLabel(self.view_point_frame)
        self.size_of_frame_lbl.setGeometry(QtCore.QRect(20, 95, 250, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.size_of_frame_lbl.setFont(font)
        self.size_of_frame_lbl.setObjectName("Size_of_frame_lbl")

        self.line_2 = QtWidgets.QFrame(self.view_point_frame)
        self.line_2.setGeometry(QtCore.QRect(30, 120, 160, 10))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")

        self.mainframe = QtWidgets.QFrame(self.view_point_frame)
        self.mainframe.setGeometry(QtCore.QRect(0, 140, 230, 400))
        self.mainframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mainframe.setObjectName("mainframe")

        self.first_lbl = QtWidgets.QLabel(self.mainframe)
        self.first_lbl.setGeometry(QtCore.QRect(75, 0, 35, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.first_lbl.setFont(font)
        self.first_lbl.setObjectName("first_lbl")

        self.last_lbl = QtWidgets.QLabel(self.mainframe)
        self.last_lbl.setGeometry(QtCore.QRect(145, 0, 35, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.last_lbl.setFont(font)
        self.last_lbl.setObjectName("last_lbl")

        self.frame_lbl = QtWidgets.QLabel(self.mainframe)
        self.frame_lbl.setGeometry(QtCore.QRect(0, 70, 50, 25))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.frame_lbl.setFont(font)
        self.frame_lbl.setObjectName("frame_lbl")

        self.col_lbl = QtWidgets.QLabel(self.mainframe)
        self.col_lbl.setGeometry(QtCore.QRect(0, 30, 70, 25))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.col_lbl.setFont(font)
        self.col_lbl.setObjectName("col_lbl")

        self.line = QtWidgets.QFrame(self.mainframe)
        self.line.setGeometry(QtCore.QRect(125, 5, 3, 100))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.first_frame_txt = QtWidgets.QLineEdit(self.mainframe)
        self.first_frame_txt.setGeometry(QtCore.QRect(65, 70, 50, 30))
        self.first_frame_txt.setObjectName("first_frame_txt")

        self.first_col_txt = QtWidgets.QLineEdit(self.mainframe)
        self.first_col_txt.setGeometry(QtCore.QRect(65, 30, 50, 30))
        self.first_col_txt.setObjectName("first_col_txt")

        self.last_frame_txt = QtWidgets.QLineEdit(self.mainframe)
        self.last_frame_txt.setGeometry(QtCore.QRect(135, 70, 50, 30))
        self.last_frame_txt.setObjectName("last_frame_txt")

        self.last_col_txt = QtWidgets.QLineEdit(self.mainframe)
        self.last_col_txt.setGeometry(QtCore.QRect(135, 30, 50, 30))
        self.last_col_txt.setObjectName("last_col_txt")

        self.step_lbl = QtWidgets.QLabel(self.mainframe)
        self.step_lbl.setGeometry(QtCore.QRect(155, 118, 100, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.step_lbl.setFont(font)
        self.step_lbl.setObjectName("step_lbl")

        self.left_right_spin = QtWidgets.QDoubleSpinBox(self.mainframe)
        self.left_right_spin.setGeometry(QtCore.QRect(90, 180, 60, 30))
        self.left_right_spin.setSingleStep(1)
        self.left_right_spin.setObjectName("left_right_spin")

        self.forward_back_spin = QtWidgets.QSpinBox(self.mainframe)
        self.forward_back_spin.setGeometry(QtCore.QRect(90, 140, 60, 30))
        self.forward_back_spin.setObjectName("forward_back_spin")

        self.angle_spin = QtWidgets.QDoubleSpinBox(self.mainframe)
        self.angle_spin.setGeometry(QtCore.QRect(90, 220, 60, 30))
        self.angle_spin.setMaximum(180.0)
        self.angle_spin.setObjectName("angle_spin")

        self.left_right_step = QtWidgets.QLineEdit(self.mainframe)
        self.left_right_step.setGeometry(QtCore.QRect(160, 180, 60, 30))
        self.left_right_step.setObjectName("lrstep")

        self.forward_back_step = QtWidgets.QLineEdit(self.mainframe)
        self.forward_back_step.setGeometry(QtCore.QRect(160, 140, 60, 30))
        self.forward_back_step.setObjectName("forward_back_step")

        self.angle_step = QtWidgets.QLineEdit(self.mainframe)
        self.angle_step.setGeometry(QtCore.QRect(160, 220, 60, 30))
        self.angle_step.setObjectName("angle_step")

        self.left_right_lbl = QtWidgets.QLabel(self.mainframe)
        self.left_right_lbl.setGeometry(QtCore.QRect(0, 185, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.left_right_lbl.setFont(font)
        self.left_right_lbl.setObjectName("lrlbl")

        self.forward_back_lbl = QtWidgets.QLabel(self.mainframe)
        self.forward_back_lbl.setGeometry(QtCore.QRect(0, 130, 81, 45))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.forward_back_lbl.setFont(font)
        self.forward_back_lbl.setObjectName("forward_back_lbl")

        self.angle_lbl = QtWidgets.QLabel(self.mainframe)
        self.angle_lbl.setGeometry(QtCore.QRect(0, 220, 71, 25))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.angle_lbl.setFont(font)
        self.angle_lbl.setObjectName("angle_lbl")

        self.tabWidget.addTab(self.view_point_tab, "")
        self.refocus_tab = QtWidgets.QWidget()
        self.refocus_tab.setObjectName("refocus_tab")
        self.tabWidget.addTab(self.refocus_tab, "")

        self.refocus_view = QtWidgets.QLabel(self.refocus_tab)
        self.refocus_view.setGeometry(QtCore.QRect(250, 10, 340, 430))
        self.refocus_view.setText("")
        self.refocus_view.setScaledContents(True)
        self.refocus_view.setObjectName("refocus_view")

        self.focus_frame = QtWidgets.QFrame(self.refocus_tab)
        self.focus_frame.setGeometry(QtCore.QRect(10, 10, 230, 400))
        self.focus_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.focus_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.focus_frame.setObjectName("frame_2")

        self.refocus_lbl = QtWidgets.QLabel(self.focus_frame)
        self.refocus_lbl.setGeometry(QtCore.QRect(30, 10, 200, 40))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.refocus_lbl.setFont(font)
        self.refocus_lbl.setObjectName("label_6")

        self.num_of_frames_lbl_2 = QtWidgets.QLabel(self.focus_frame)
        self.num_of_frames_lbl_2.setGeometry(QtCore.QRect(20, 65, 250, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.num_of_frames_lbl_2.setFont(font)
        self.num_of_frames_lbl_2.setObjectName("num_of_frames_lbl_2")

        self.size_of_frame_lbl_2 = QtWidgets.QLabel(self.focus_frame)
        self.size_of_frame_lbl_2.setGeometry(QtCore.QRect(20, 95, 250, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.size_of_frame_lbl_2.setFont(font)
        self.size_of_frame_lbl_2.setObjectName("size_of_frame_lbl_2")

        self.line_3 = QtWidgets.QFrame(self.focus_frame)
        self.line_3.setGeometry(QtCore.QRect(30, 120, 160, 10))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")

        self.near_lbl = QtWidgets.QLabel(self.focus_frame)
        self.near_lbl.setGeometry(QtCore.QRect(95, 175, 35, 15))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.near_lbl.setFont(font)
        self.near_lbl.setObjectName("near_lbl")

        self.far_lbl = QtWidgets.QLabel(self.focus_frame)
        self.far_lbl.setGeometry(QtCore.QRect(100, 260, 35, 15))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.far_lbl.setFont(font)
        self.far_lbl.setObjectName("far_lbl")

        self.step_lbl_2 = QtWidgets.QLabel(self.focus_frame)
        self.step_lbl_2.setGeometry(QtCore.QRect(16, 300, 100, 20))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.step_lbl_2.setFont(font)
        self.step_lbl_2.setObjectName("step_lbl_2")

        self.refocus_step = QtWidgets.QLineEdit(self.focus_frame)
        self.refocus_step.setGeometry(QtCore.QRect(90, 300, 60, 30))
        self.refocus_step.setObjectName("farnearstep")

        self.refocus_spin = QtWidgets.QDoubleSpinBox(self.focus_frame)
        self.refocus_spin.setGeometry(QtCore.QRect(80, 200, 70, 50))
        self.refocus_spin.setSingleStep(1)
        self.refocus_spin.setObjectName("farnearspin")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 622, 18))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslate_ui(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslate_ui(self, MainWindow):
        self._translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(self._translate("MainWindow", "MainWindow"))
        self.nums_of_frames_lbl.setText(self._translate("MainWindow", "Number of frames: 0"))
        self.left_right_lbl.setText(self._translate("MainWindow", "left/right"))
        self.forward_back_lbl.setText(self._translate("MainWindow", "forward/\nbackward"))
        self.angle_lbl.setText(self._translate("MainWindow", "angle"))
        self.step_lbl.setText(self._translate("MainWindow", "step size"))
        self.last_lbl.setText(self._translate("MainWindow", "last"))
        self.first_lbl.setText(self._translate("MainWindow", "first"))
        self.frame_lbl.setText(self._translate("MainWindow", "frame"))
        self.col_lbl.setText(self._translate("MainWindow", "column"))
        self.size_of_frame_lbl.setText(self._translate("MainWindow", "Size of frame: 0X0"))
        self.view_point_lbl.setText(self._translate("MainWindow", "View Point"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.view_point_tab), self._translate("MainWindow", "View Point"))
        self.near_lbl.setText(self._translate("MainWindow", "near"))
        self.far_lbl.setText(self._translate("MainWindow", "far"))
        self.refocus_lbl.setText(self._translate("MainWindow", "ReFocusing"))
        self.num_of_frames_lbl_2.setText(self._translate("MainWindow", "Number of frames: 0"))
        self.size_of_frame_lbl_2.setText(self._translate("MainWindow", "Size of frame: 0X0"))
        self.step_lbl_2.setText(self._translate("MainWindow", "step size"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.refocus_tab), self._translate("MainWindow", "Refocusing"))
        self.tabWidget.blockSignals(True)
        self.tabWidget.currentChanged.connect(self.change_tab)
        self.tabWidget.blockSignals(False)
        self.menuFile.setTitle(self._translate("MainWindow", "New sequence"))
        self.actionOpen.setText(self._translate("MainWindow", "Load new sequence"))
        self.menuFile.actions()[0].triggered.connect(self.open_video)
        self.left_right_spin.setDisabled(True)
        self.angle_spin.setDisabled(True)
        self.forward_back_spin.setDisabled(True)
        self.left_right_step.setDisabled(True)
        self.angle_step.setDisabled(True)
        self.forward_back_step.setDisabled(True)
        self.first_frame_txt.setDisabled(True)
        self.last_frame_txt.setDisabled(True)
        self.first_col_txt.setDisabled(True)
        self.last_col_txt.setDisabled(True)
        self.refocus_spin.setDisabled(True)
        self.refocus_step.setDisabled(True)
        self.left_right_spin.valueChanged.connect(self.change_column)
        self.forward_back_spin.valueChanged.connect(self.change_frame)
        self.angle_spin.valueChanged.connect(self.change_angle)
        self.left_right_step.textChanged.connect(lambda event: self.left_right_spin.setSingleStep(float(self.left_right_step.text())))
        self.forward_back_step.textChanged.connect(lambda event: self.forward_back_spin.setSingleStep(int(self.forward_back_step.text())))
        self.angle_step.textChanged.connect(lambda event: self.angle_spin.setSingleStep(float(self.angle_step.text())))
        self.first_frame_txt.textChanged.connect(self.value_change)
        self.last_frame_txt.textChanged.connect(self.value_change)
        self.first_col_txt.textChanged.connect(self.value_change)
        self.last_col_txt.textChanged.connect(self.value_change)
        self.refocus_step.textChanged.connect(lambda event: self.refocus_spin.setSingleStep(float(self.refocus_step.text())))
        self.refocus_spin.valueChanged.connect(self.change_image_refocus)
        self.left_right_step.setText(self._translate("MainWindow", "1.0"))
        self.forward_back_step.setText(self._translate("MainWindow", "1"))
        self.angle_step.setText(self._translate("MainWindow", "0.5"))

        for i, p in enumerate(self.all_frames.paths):
            self.menuFile.addAction(os.path.basename(p))
            self.menuFile.actions()[i + 1].triggered.connect(lambda checked, path=p: self.change_video(path))

    def open_video(self):
        """
        open new sequence
        :return:
        """
        dir_path = str(QFileDialog.getExistingDirectory(self, 'Please select a directory of frames'))
        if dir_path not in self.all_frames.paths:
            with open(self.all_frames.data_path, 'a') as data_file:
                data_file.write(dir_path + '\n')
            self.all_frames.paths.append(dir_path)
            self.menuFile.addAction(os.path.basename(dir_path))
            self.all_frames.read_and_fix_images(dir_path)
            self.menuFile.actions()[len(self.all_frames.paths)].triggered.connect \
                (lambda checked, p=dir_path: self.change_video(p))
        self.change_video(dir_path)

    def change_video(self, dir_path):
        video_data = os.path.join(dir_path, self.all_frames.video_data_path)
        if self.tabWidget.currentIndex() == 0:
            self.fixed_frames_view_point, self.dx_motions_vp = np.load(video_data, allow_pickle=True)
            height, width = self.fixed_frames_view_point[0].shape[0], self.fixed_frames_view_point[0].shape[1]
            self.left_right_spin.setEnabled(True)
            self.forward_back_spin.setEnabled(True)
            self.angle_spin.setEnabled(True)
            self.left_right_step.setEnabled(True)
            self.forward_back_step.setEnabled(True)
            self.angle_step.setEnabled(True)
            self.first_frame_txt.setEnabled(True)
            self.last_frame_txt.setEnabled(True)
            self.first_col_txt.setEnabled(True)
            self.last_col_txt.setEnabled(True)
            self.nums_of_frames_lbl.setText(
                self._translate("MainWindow", "Number of frames: " + str(len(self.fixed_frames_view_point))))
            self.size_of_frame_lbl.setText(self._translate("MainWindow", "Size of frame: " + str(width) + "X" + str(height)))

            self.first_frame_txt.setText('0')
            self.last_frame_txt.setText(str(len(self.fixed_frames_view_point) - 1))

            self.first_col_txt.setText('0.0')
            self.last_col_txt.setText('0.0')
            self.left_right_step.setText('1.0')
            self.forward_back_step.setText('1')
            self.angle_step.setText('1.0')
            self.left_right_prev_val = 0
            self.forward_back_prev_val = 0
            self.angle_prev_val = 0
            self.left_right_spin.setSingleStep(1)
            self.left_right_spin.setValue(0)
            self.forward_back_spin.setSingleStep(1)
            self.forward_back_spin.setValue(0)
            self.angle_spin.setValue(90)
            self.left_right_spin.setMaximum(width - 1)
            self.forward_back_spin.setMaximum(len(self.fixed_frames_view_point) - 1)
            self.change_image_view_point()

        else:
            self.fixed_frames_refocus, self.dx_motions_refocus = np.load(video_data, allow_pickle=True)
            height, width = self.fixed_frames_refocus[0].shape[0], self.fixed_frames_refocus[0].shape[1]
            self.num_of_frames_lbl_2.setText(
                self._translate("MainWindow", "Number of frames: " + str(len(self.fixed_frames_refocus))))
            self.size_of_frame_lbl_2.setText(self._translate("MainWindow", "Size of frame: " + str(width) + "X" + str(height)))
            self.refocus_spin.setEnabled(True)
            self.refocus_step.setEnabled(True)
            self.refocus_spin.setMaximum(np.inf)
            self.refocus_spin.setMinimum(-np.inf)
            self.refocus_step.setText('1.0')
            self.refocus_spin.setSingleStep(1)
            self.refocus_spin.setValue(0)
            self.change_image_refocus()

    def change_image(self, im):
        height, width, channels = im.shape
        bytes_per_line = channels * width
        q_im = QImage(im.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap1 = QtGui.QPixmap.fromImage(q_im)
        pixmap = QtGui.QPixmap(pixmap1)
        # check which tab is in focus
        if self.tabWidget.currentIndex() == 0:
            self.view_point_im = pixmap
        else:
            self.refocus_im = pixmap
        self.change_tab()

    def change_image_refocus(self):
        im = self.all_frames.refocusing(self.fixed_frames_refocus, self.dx_motions_refocus, float(self.refocus_spin.text()))
        if im is not None:
            self.change_image(im)

    def change_image_view_point(self):
        im = self.all_frames.change_view_point(self.fixed_frames_view_point, self.dx_motions_vp,
                                               int(self.first_frame_txt.text()), int(float(self.first_col_txt.text())),
                                               int(self.last_frame_txt.text()), int(float(self.last_col_txt.text())))
        if im is not None:
            self.change_image(im)

    def change_tab(self):
        if self.tabWidget.currentIndex() == 0:
            if self.view_point_im:
                self.view_point_view.setText("")
                self.view_point_view.setPixmap(self.view_point_im)
                self.view_point_view.resize(self.view_point_im.width(), self.view_point_im.height())
                self.mainwindow.resize(270 + self.view_point_im.width(), max(550, 60 + self.view_point_im.height()))
                self.tabWidget.resize(260 + self.view_point_im.width(), max(490, self.view_point_im.height()))
            else:
                self.mainwindow.resize(250, 550)
                self.tabWidget.resize(250, 490)
        else:
            if self.refocus_im:
                self.refocus_view.setText("")
                self.refocus_view.setPixmap(self.refocus_im)
                self.refocus_view.resize(self.refocus_im.width(), self.refocus_im.height())
                self.mainwindow.resize(270 + self.refocus_im.width(), max(550, 60 + self.refocus_im.height()))
                self.tabWidget.resize(260 + self.refocus_im.width(), max(490, self.refocus_im.height()))
            else:
                self.mainwindow.resize(250, 550)
                self.tabWidget.resize(250, 490)

    def change_column(self):
        if not self.timer_flag:
            if float(self.left_right_prev_val) < float(self.left_right_spin.text()):
                first_col_val = float(self.first_col_txt.text()) + self.left_right_spin.singleStep()
                last_col_val = float(self.last_col_txt.text()) + self.left_right_spin.singleStep()
            else:
                first_col_val = float(self.first_col_txt.text()) - self.left_right_spin.singleStep()
                last_col_val = float(self.last_col_txt.text()) - self.left_right_spin.singleStep()
            if first_col_val < 0:
                self.first_col_txt.setText('0.0')
            elif first_col_val > self.left_right_spin.maximum():
                self.first_col_txt.setText(str(self.left_right_spin.maximum()))
            else:
                self.first_col_txt.setText(str(first_col_val))
            if last_col_val < 0:
                self.last_col_txt.setText('0.0')
            elif last_col_val > self.left_right_spin.maximum():
                self.last_col_txt.setText(str(self.left_right_spin.maximum()))
            else:
                self.last_col_txt.setText(str(last_col_val))
        self.left_right_prev_val = self.left_right_spin.text()
        self.change_image_view_point()

    def change_frame(self):
        if not self.timer_flag:
            if int(self.forward_back_prev_val) < int(self.forward_back_spin.text()):
                first_frame_val = int(self.first_frame_txt.text()) + self.forward_back_spin.singleStep()
                last_frame_val = int(self.last_frame_txt.text()) + self.forward_back_spin.singleStep()
            else:
                first_frame_val = int(self.first_frame_txt.text()) - self.forward_back_spin.singleStep()
                last_frame_val = int(self.last_frame_txt.text()) - self.forward_back_spin.singleStep()
            if first_frame_val < 0:
                self.first_frame_txt.setText('0')
            elif first_frame_val > self.forward_back_spin.maximum():
                self.first_frame_txt.setText(str(self.forward_back_spin.maximum()))
            else:
                self.first_frame_txt.setText(str(first_frame_val))
            if last_frame_val < 0:
                self.last_frame_txt.setText('0')
            elif last_frame_val > self.forward_back_spin.maximum():
                self.last_frame_txt.setText(str(self.forward_back_spin.maximum()))
            else:
                self.last_frame_txt.setText(str(last_frame_val))
        # self.manual_flag = False
        self.forward_back_prev_val = self.forward_back_spin.text()
        self.change_image_view_point()

    def calculate_new_angle(self):
        if float(self.first_col_txt.text()) == float(self.last_col_txt.text()):
            angle = 90
        else:
            angle = np.arctan(((int(self.last_frame_txt.text()) - int(self.first_frame_txt.text())) /
                               (float(self.last_col_txt.text()) - float(self.first_col_txt.text())))) * (180 / np.pi)
        if angle < 0:
            angle += 180
        return angle

    def change_angle(self):
        angle = float(self.angle_spin.text())
        if angle > 90:
            angle -= 180
        mid_col = (float(self.last_col_txt.text()) + float(self.first_col_txt.text())) / 2
        slope = np.tan(angle * np.pi / 180)
        width = self.fixed_frames_view_point[0].shape[1]
        mid_frame = (int(self.last_frame_txt.text()) - int(self.first_frame_txt.text())) // 2
        if angle == 90:
            self.last_col_txt.setText(str(mid_col))
            self.first_col_txt.setText(str(mid_col))
        elif slope == 0:
            self.first_col_txt.setText('0.0')
            self.last_col_txt.setText(str(float(width - 1)))
            self.first_frame_txt.setText(str(mid_frame))
            self.last_frame_txt.setText(str(mid_frame))
        else:
            b = mid_frame - slope * mid_col
            first_col = (int(self.first_frame_txt.text()) - b) / slope
            last_col = (int(self.last_frame_txt.text()) - b) / slope
            if first_col < 0:
                first_col = 0.0
                self.first_frame_txt.setText(str(int(slope * first_col + b)))
            elif first_col >= width:
                first_col = width - 1
                self.first_frame_txt.setText(str(int(slope * width + b)))
            if last_col < 0:
                last_col = 0.0
                self.last_frame_txt.setText(str(int(b)))
            elif last_col >= width:
                last_col = width - 1
                self.last_frame_txt.setText(str(int(slope * width + b)))
            self.first_col_txt.setText(str(np.round(float(first_col), 2)))
            self.last_col_txt.setText(str(np.round(float(last_col), 2)))

    def value_change(self):
        if self.timer_id != -1:
            self.killTimer(self.timer_id)

        self.timer_id = self.startTimer(500)

    def timerEvent(self, event):
        try:
            self.killTimer(self.timer_id)
            self.timer_id = -1
            self.timer_flag = True
            self.left_right_spin.setValue(float(self.first_col_txt.text()))
            self.forward_back_spin.setValue(int(self.first_frame_txt.text()))
            self.first_col_txt.setText(str(min(float(self.first_col_txt.text()), self.left_right_spin.maximum())))
            self.first_frame_txt.setText(str(min(int(self.first_frame_txt.text()), self.forward_back_spin.maximum())))
            self.last_col_txt.setText(str(min(float(self.last_col_txt.text()), self.left_right_spin.maximum())))
            self.last_frame_txt.setText(str(min(int(self.last_frame_txt.text()), self.forward_back_spin.maximum())))
            self.angle_spin.setValue(float(self.calculate_new_angle()))
            self.change_image_view_point()
            self.timer_flag = False
        except ValueError as E:
            return


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setup_ui(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())