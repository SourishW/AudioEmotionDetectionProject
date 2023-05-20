from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import os
import random as rd
from pyaudio_api_testing import Record

class GrapherTool(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(GrapherTool, self).__init__(*args, **kwargs)
        self.__graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.__graphWidget)

        sampleN = Record.BUFSIZE
        self.__x = list(range(sampleN)) 
        self.__data = [rd.randint(0,100) for junk in range(sampleN)] 
        self.__graphWidget.setBackground('b')

        pen = pg.mkPen(color=(255, 100, 50))
        self.__data_line =  self.__graphWidget.plot(self.__x, self.__data, pen=pen)

        self.__timer = QtCore.QTimer()
        self.__timer.setInterval(1)
        self.__timer.timeout.connect(self.__update_plot)
        self.__timer.start()

    def __update_plot(self):
        self.__data_line.setData(self.__x, self.__data)

    def update_data(self, data):
        self.data = data


