import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsWidget, QGraphicsLinearLayout
import pyqtgraph as pg

class DoublePlotWindow(QMainWindow):
    def __init__(self):
        super(DoublePlotWindow, self).__init__()

        # Create the top plot
        plot_top = pg.PlotWidget()

        # Create the bottom plot
        plot_bottom = pg.PlotWidget()

        # Set the layout
        layout = QGraphicsWidget()
        linear_layout = QGraphicsLinearLayout()
        linear_layout.setOrientation(pg.QtCore.Qt.Vertical)
        layout.setLayout(linear_layout)

        # Create QGraphicsWidget items for the plots
        top_widget = QGraphicsWidget()
        top_widget.setLayout(pg.GraphicsLayout())
        top_widget.layout.addItem(plot_top)

        bottom_widget = QGraphicsWidget()
        bottom_widget.setLayout(pg.GraphicsLayout())
        bottom_widget.layout.addItem(plot_bottom)

        # Add the items to the layout
        linear_layout.addItem(top_widget)
        linear_layout.addItem(bottom_widget)

        # Set the layout as the central widget
        self.setCentralWidget(layout)

        # Set some properties for the plots
        plot_top.plot([1, 2, 3, 4, 5], pen='r', symbol='o')
        plot_bottom.plot([5, 4, 3, 2, 1], pen='b', symbol='x')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DoublePlotWindow()
    window.show()
    sys.exit(app.exec_())
