import numpy as np
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QResizeEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random as rd

WHITE = (1.0, 1.0, 1.0)  # white
DARK_GRAY = "#161616"


class Plotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("16 bit signed real-time audio signal")
        self.setGeometry(100, 100, 1600, 1000)

        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(211)  # First subplot for samples vs amplitude
        self.ax2 = self.figure.add_subplot(212)  # Second subplot for frequency vs amplitude
    
        self.ax1.set_ylabel(
            "Amplitude (Signed Integer)",
            color="white",
            weight="bold",
            fontsize=12,
        )
        self.ax1.set_xlabel(
            "Time (Samples)",
            color="white",
            weight="bold",
            fontsize=12,
        )
        self.ax1.set_title(
            "Time Domain", color="white", weight="bold", fontsize=16
        )

        self.ax2.set_ylabel(
            "Amplitude (magnitude)",
            color="white",
            weight="bold",
            fontsize=12,
        )
        self.ax2.set_xlabel(
            "Frequency (Hz)",
            color="white",
            weight="bold",
            fontsize=12,
        )
        self.ax2.set_title(
            "Fourier Domain", color="white", weight="bold", fontsize=16
        )

        low_end = -2 ** 15 - 1
        high_end = 2 ** 15
        self.ax1.set_ylim(low_end, high_end)
        self.ax1.set_yticks([low_end, low_end / 3, 0, high_end / 3, high_end])
  

        # Set dark mode color scheme
        self.figure.patch.set_facecolor(DARK_GRAY)  # Set background color
        self.ax1.set_facecolor("#282828")  # Set axes background color
        self.ax2.set_facecolor("#282828")  # Set axes background color

        self.ax1.tick_params(colors=WHITE)  # Set tick color
        self.ax2.tick_params(colors=WHITE)  # Set tick color
        for elem in (
            self.ax1.spines["bottom"],
            self.ax1.spines["left"],
            self.ax1.xaxis.label,
            self.ax1.yaxis.label,
        ):
            elem.set_color(WHITE)
        for elem in (
            self.ax2.spines["bottom"],
            self.ax2.spines["left"],
            self.ax2.xaxis.label,
            self.ax2.yaxis.label,
        ):
            elem.set_color(WHITE)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.line1, = self.ax1.plot([], [], lw=2)  # Line for the first plot
        self.line2, = self.ax2.plot([], [], lw=2)  # Line for the second plot

        self.maxima = 0

        self.show()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.canvas.resize(event.size().width(), event.size().height())

    def update_sample_plot(self, buffer):
        x = np.arange(0, len(buffer))
        self.line1.set_data(x, buffer)

        self.ax1.relim()
        self.ax1.autoscale_view(scalex=True, scaley=True)

        self.canvas.draw()
        self.canvas.flush_events()

    def update_fourier_plot(self, frequencies, amplitudes):
        if rd.random()<0.05:
            self.maxima/=2.0
        if (np.max(amplitudes) > self.maxima):
            self.maxima = np.max(amplitudes)/1.5
            high_end = self.maxima
            self.ax2.set_ylim(0, high_end)
            nticks = 10
            self.ax2.set_yticks([(i * high_end) / nticks for i in range(nticks)])
        
        self.line2.set_data(frequencies, amplitudes)
        self.ax2.relim()
        self.ax2.set_xscale('log')
        self.ax2.autoscale_view(scalex=True, scaley=True)

        self.canvas.draw()
        self.canvas.flush_events()
