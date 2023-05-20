import numpy as np
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QResizeEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

WHITE = (1.0, 1.0, 1.0) # light orange

class Plotter(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("16 bit signed real-time audio signal")
        self.setGeometry(100, 100, 800, 800)

        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.ax.set_ylabel("Amplitude", color=WHITE)  # Set y-axis label color
        self.ax.set_xlabel("Time", color=WHITE)  # Set x-axis label color

        low_end = -2 ** 15 - 1
        high_end = 2 ** 15
        self.ax.set_ylim(low_end, high_end)
        self.ax.set_yticks([low_end, low_end / 2, 0, high_end / 2, high_end])

        # Set dark mode color scheme
        self.figure.patch.set_facecolor('#363636')  # Set background color
        self.ax.set_facecolor('#282828')  # Set axes background color
        self.ax.tick_params(colors=WHITE)  # Set tick color
        self.ax.spines['bottom'].set_color(WHITE)  # Set bottom spine color
        self.ax.spines['left'].set_color(WHITE)  # Set left spine color
        self.ax.xaxis.label.set_color(WHITE)  # Set x-axis label color
        self.ax.yaxis.label.set_color(WHITE)  # Set y-axis label color

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.line, = self.ax.plot([], [], lw=2)

        self.show()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.canvas.resize(event.size().width(), event.size().height())

    def update_plot(self, buffer):
        x = np.arange(0, len(buffer))
        self.line.set_data(x, buffer)

        self.ax.relim()
        self.ax.autoscale_view(scalex=True, scaley=True)

        self.canvas.draw()
        self.canvas.flush_events()
