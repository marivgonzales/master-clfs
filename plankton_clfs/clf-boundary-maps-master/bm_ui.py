from PyQt5.QtWidgets import QMainWindow, QLabel, QSizePolicy, QSplitter
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtWidgets import QAction, QActionGroup
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QFileDialog, QSpinBox
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QFrame
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import hsv_to_rgb

import numpy as np
import sys
import json
import joblib

from sklearn.neighbors import NearestNeighbors

import lamp
from utils import SampleSquareUniform, LoadProjection, HSV2RGB, Lerp
from utils import bmap_to_categorical
from utils import bmap_to_categorical_base

from boundarymap import Grid
from nninv import NNInv
from lamp import ILAMP
from lamp import RBFInv


# TODO:
# 1. Make sample suggestions selectable
# 2. Make it possible to set the label of a sample
# 3. Click and stop the brushing
# 4. Show nearest decision boundary
# 5. Construct more data files (densemap, projection, predicted labels)

# Nearest decision boundary:
#   - Distance to 2D Boundary:
#   Perform a BFS from each pixel stopping when a sample with different hue
#   value is found. Consider CMAP_ORIG and CMAP_SYN.
#   - Distance to nD Boundary:
#   Consider the original dataset augmented by iLAMP samples. Save iLAMP
#   synthetic imgs along with the label the classifier assigns to them. Now,
#   for each pixel, check one of it's iLAMP synthetic sample (or real sample if
#   exists). Get the nearest nD point to it that has a different label. Compute
#   distance.

# number of neighbors to visualize
NUM_NEIGHBORS = 5
NUM_SAMPLES = 16
IMG_SIZE = 28

# TODO: read from file
GRID_SIZE = 200


SYN_HUES = np.array([216, 18, 126, 306, 270, 90, 198, 342, 54, 162])/360.0

COLORS_HUE = np.array([[0.09, 0.414, 0.9, 1.],
                       [0.9, 0.333, 0.09, 1.],
                       [0.09, 0.9, 0.171, 1.],
                       [0.9, 0.09, 0.819, 1.],
                       [0.495, 0.09, 0.9, 1.],
                       [0.495, 0.9, 0.09, 1.],
                       [0.09, 0.657, 0.9, 1.],
                       [0.9, 0.09, 0.333, 1.],
                       [0.9, 0.819, 0.09, 1.],
                       [0.09, 0.9, 0.657, 1.]])

COLORS_CAT = np.array([[166, 206, 227, 255],
                       [31,  120, 180, 255],
                       [178, 223, 138, 255],
                       [51,  160,  44, 255],
                       [251, 154, 153, 255],
                       [227, 26,   28, 255],
                       [253, 191, 111, 255],
                       [255, 127,   0, 255],
                       [202, 178, 214, 255],
                       [106, 61,  154, 255]])/255.0

COLORS_CAT2 = np.array([[166, 206, 227, 255],
                        [31,  120, 180, 255],
                        [178, 223, 138, 255],
                        [51,  160,  44, 255],
                        [251, 154, 153, 255],
                        [227, 26,   28, 255],
                        [253, 191, 111, 255],
                        [255, 127,   0, 255],
                        [202, 178, 214, 255],
                        [106, 61,  154, 255],
                        [127, 127, 127, 127],
                        [127, 127, 127, 127],
                        [127, 127, 127, 127],
                        [127, 127, 127, 127],
                        [127, 127, 127, 127],
                        [127, 127, 127, 127],
                        [127, 127, 127, 127],
                        [127, 127, 127, 127],
                        [127, 127, 127, 127],
                        [127, 127, 127, 127]])/255.0

COLORS = COLORS_HUE
hsv2rgb = hsv_to_rgb

# COLORS = COLORS_CAT
# hsv2rgb = bmap_to_categorical

# COLORS = COLORS_CAT
# hsv2rgb = bmap_to_categorical_base

CMAP_ORIG = np.array([234, 0, 108, 288, 252, 72, 180, 324, 36, 144])/360.0
CMAP_SYN = np.array([216, 18, 126, 306, 270, 90, 198, 342, 54, 162])/360.0


def transform_db_map(dist_map_in, db_map_in):
    dist_map = np.copy(dist_map_in)
    dist_map /= dist_map.max()
    dist_map = 1.0 - dist_map

    db_map = np.copy(db_map_in)
    # db_map[:, :, 1] = (1.0 - dist_map)*db_map[:, :, 1] + dist_map*(np.maximum(db_map[:, :, 1] - 0.2, 0.0))

    # db_map[:, :, 2] = 0.1 + 0.9*(dist_map**(1.0))
    N = 4
    db_map[:, :, 2] = 0.1 + 0.9*((dist_map%(1.0/N))*N)
    return db_map


# zoom and pan code adapted from:
# https://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel/12793033
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_axis_off()
        self.axes.set_aspect('equal')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.init_evt_state()

        self.pt_size = 20

        self.scatter_handler = None
        self.lines = None
        self.boundary_map = None

        line_color = [0.0, 0.0, 0.0, 0.5]
        self.lines = mlines.Line2D([], [], color=line_color)
        self.axes.add_line(self.lines)

        # self.mpl_connect('resize_event', self.on_resize)

    def init_evt_state(self):
        self.press = False
        self.cur_xlim = None
        self.cur_ylim = None
        self.xpress = None
        self.ypress = None

    # def on_resize(self, event):
    #     self.background = self.copy_from_bbox(self.axes.bbox)

    def set_scene(self, img, pts, colors):
        # self.axes.clear()
        self.img = img
        self.pts = pts
        self.colors = colors

        self.plot_proj(self.pts, self.colors)
        self.plot_clf_map(self.img)
        # self.background = self.copy_from_bbox(self.axes.bbox)

    def plot_proj(self, pts, colors):
        # FIXME if handler is not none, just update it without calling
        # scatter funcion again. Problem: blit() was not working
        if self.scatter_handler is not None:
            self.scatter_handler.remove()
            self.scatter_handler = None

        if pts is None:
            self.draw()
            return

        num_pts = pts.shape[0]
        sizes = np.array([self.pt_size]*num_pts)
        self.scatter_handler = self.axes.scatter(pts[:, 0], pts[:, 1], s=sizes,
                                                 color=colors, picker=1,
                                                 linewidths=0)
        self.draw()

    def plot_clf_map(self, img):
        if self.boundary_map is not None:
            self.boundary_map.remove()
            self.boundary_map = None
        extent = (0.0, img.shape[0], 0.0, img.shape[1])
        self.boundary_map = self.axes.imshow(img, interpolation='None',
                                             origin='lower', extent=extent)
        self.draw()

    # Draws lines from src coords to every point in dests list
    def draw_lines_to_pts(self, src, dests):
        xlines = []
        ylines = []

        for dest in dests:
            xlines.append(src[0])
            ylines.append(src[1])

            xlines.append(dest[0])
            ylines.append(dest[1])

        # FIXME: blit not working
        # self.lines.set_data([], [])
        # self.restore_region(self.background)
        # self.axes.draw_artist(self.lines)
        # self.blit(self.axes.bbox)

        # self.lines.set_data(xlines, ylines)
        # self.restore_region(self.background)
        # self.axes.draw_artist(self.lines)
        # self.blit(self.axes.bbox)

        self.lines.set_data(xlines, ylines)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(300, 300, 1400, 900)
        self.setWindowTitle('test')

        self.clfData = CLFData()
        # self.clfData.FromJSON("data/fm/fm.json")

        self.highlightNeighbors = False
        self.freeze = False
        self.show_closest_label = False

        # ALL = 0
        # Miss = 1
        # None = 2
        self.show_pts_sel = 0

        self.newsamplesUI = NewSamplesUI()

        self.inv_proj_samples = [None for _ in range(NUM_SAMPLES)]
        self.ilamp_radius = 5.0/GRID_SIZE
        self.nd_radius = 30.0

        self.new_samples = []
        self.new_samples_y = []
        # self.newsamplesUI.add_sample_btn[0].clicked.connect(self.save0)
        # self.newsamplesUI.add_sample_btn[1].clicked.connect(self.save1)
        # self.newsamplesUI.add_sample_btn[2].clicked.connect(self.save2)
        # self.newsamplesUI.add_sample_btn[3].clicked.connect(self.save3)
        # self.newsamplesUI.add_sample_btn[4].clicked.connect(self.save4)
        self.initUI()

        self.press_cid = None
        self.release_cid = None
        self.motion_cid = None
        self.scroll_cid = None
        self.key_release_cid = None

        self.show_distmap = False
        self.show_distmap_nd = False
        self.show_distmap_nd2 = False
        self.show_distmap_nd3 = False

        self.BMAP = 0
        self.DMAP2D = 1
        self.DMAPND = 2
        self.DMAPND2 = 3
        self.DMAPND3 = 4

        self.features = False

    def save0(self):
        self.save(0)

    def save1(self):
        self.save(1)

    def save2(self):
        self.save(2)

    def save3(self):
        self.save(3)

    def save4(self):
        self.save(4)

    def save(self, i):
        y = self.newsamplesUI.ilamp_labels[i].value()
        self.new_samples_y.append(y)

        # img = self.newsamplesUI.syntheticImgs[i].pixmap().toImage()
        # bits = img.bits()
        # bits.setsize(28*28)
        # arr = np.frombuffer(bits, np.uint8).reshape((28, 28, 1))
        # arr = np.ndarray(shape=(28, 28), buffer=bits, dtype=np.uint8)
        # s = img.bits().asstring(28*28*1)
        # arr = np.fromstring(s, dtype=np.uint8)

        self.new_samples.append(np.copy(self.inv_proj_samples[i]))
        self.newsamplesUI.add_sample_btn[i].setCheckable(False)
        # TODO: create new sample

    def initUI(self):
        # TODO: initCanvas()?
        self.canvas = PlotCanvas(self)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()

        # self.canvas.mpl_connect('button_press_event', self.on_press)
        # self.canvas.mpl_connect('button_release_event', self.on_release)
        # self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        # self.canvas.mpl_connect('scroll_event', self.on_scroll)
        # self.canvas.mpl_connect('key_release_event', self.on_key_release)

        menuBar = self.menuBar()
        openMenu = menuBar.addMenu('Open')
        showMenu = menuBar.addMenu('Show')

        allAct = QAction("All", self)
        allAct.setCheckable(True)
        allAct.setChecked(True)
        allAct.triggered.connect(lambda: self.showPts(0))

        missAct = QAction("Miss Clf", self)
        missAct.setCheckable(True)
        missAct.setChecked(False)
        missAct.triggered.connect(lambda: self.showPts(1))

        noneAct = QAction("None", self)
        noneAct.setCheckable(True)
        noneAct.setChecked(False)
        noneAct.triggered.connect(lambda: self.showPts(2))

        showMenu.addAction(allAct)
        showMenu.addAction(missAct)
        showMenu.addAction(noneAct)

        showMenuGroup = QActionGroup(self)
        showMenuGroup.addAction(allAct)
        showMenuGroup.addAction(missAct)
        showMenuGroup.addAction(noneAct)
        showMenuGroup.setExclusive(True)

        showMenu.addSeparator()

        # highlight action
        hlAct = QAction('Highlight Neighbors', self)
        hlAct.setShortcut("Ctrl+H")
        hlAct.setCheckable(True)
        hlAct.setChecked(False)
        hlAct.triggered.connect(self.setHLNeighbors)
        showMenu.addAction(hlAct)

        freezeAct = QAction('Freeze', self)
        freezeAct.setShortcut("Ctrl+F")
        freezeAct.setCheckable(True)
        freezeAct.setChecked(False)
        freezeAct.triggered.connect(self.freeze_act)
        showMenu.addAction(freezeAct)

        boundaryMapAct = QAction('Show boundary map')
        boundaryMapAct.setShortcut("Ctrl+B")
        boundaryMapAct.setCheckable(True)
        boundaryMapAct.setChecked(True)
        boundaryMapAct.triggered.connect(lambda: self.show_map(self.BMAP))
        showMenu.addAction(boundaryMapAct)

        distMapAct = QAction('Show dist map', self)
        distMapAct.setShortcut("Ctrl+D")
        distMapAct.setCheckable(True)
        distMapAct.setChecked(False)
        distMapAct.triggered.connect(lambda: self.show_map(self.DMAP2D))
        showMenu.addAction(distMapAct)

        distMapNDAct = QAction('Show dist map nD', self)
        distMapNDAct.setCheckable(True)
        distMapNDAct.setChecked(False)
        distMapNDAct.triggered.connect(lambda: self.show_map(self.DMAPND))
        showMenu.addAction(distMapNDAct)

        distMapND2Act = QAction('Show dist map nD2', self)
        distMapND2Act.setCheckable(True)
        distMapND2Act.setChecked(False)
        distMapND2Act.triggered.connect(lambda: self.show_map(self.DMAPND2))
        showMenu.addAction(distMapND2Act)

        distMapND3Act = QAction('Show dist map nD3', self)
        distMapND3Act.setCheckable(True)
        distMapND3Act.setChecked(False)
        distMapND3Act.triggered.connect(lambda: self.show_map(self.DMAPND3))
        showMenu.addAction(distMapND3Act)

        distMenuGroup = QActionGroup(self)
        distMenuGroup.addAction(boundaryMapAct)
        distMenuGroup.addAction(distMapAct)
        distMenuGroup.addAction(distMapNDAct)
        distMenuGroup.addAction(distMapND2Act)
        distMenuGroup.addAction(distMapND3Act)

        openProjAct = QAction("Projection", self)
        openProjAct.setShortcut("Ctrl+O")
        openProjAct.triggered.connect(self.openProjection)

        saveAct = QAction('Save', self)
        saveAct.setShortcut("Ctrl+S")
        saveAct.triggered.connect(self.save_new_samples)

        openMenu.addAction(openProjAct)
        openMenu.addAction(saveAct)

        self.errUI = ErrUI()

        splitter = QSplitter(self)
        splitter.addWidget(self.errUI)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.newsamplesUI)
        self.setCentralWidget(splitter)

    def connect_evts(self):
        self.press_cid = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.release_cid = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.scroll_cid = self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.key_release_cid = self.canvas.mpl_connect('key_release_event', self.on_key_release)

    def disconnect_evts(self):
        self.canvas.mpl_disconnect(self.press_cid)
        self.canvas.mpl_disconnect(self.release_cid)
        self.canvas.mpl_disconnect(self.motion_cid)
        self.canvas.mpl_disconnect(self.scroll_cid)

        self.canvas.mpl_disconnect(self.key_release_cid)

    def freeze_act(self):
        if self.freeze is False:
            self.freeze = True
        else:
            self.freeze = False

    def show_map(self, ui_map):
        if ui_map == self.BMAP:
            self.canvas.plot_clf_map(self.clfData.dmap)
        elif ui_map == self.DMAP2D:
            db_map_new = transform_db_map(self.clfData.dist_2d,
                                          self.clfData.dmap_hsv)
            self.canvas.plot_clf_map(hsv2rgb(db_map_new))
        elif ui_map == self.DMAPND:
            db_map_new = transform_db_map(self.clfData.dist_nd,
                                          self.clfData.dmap_hsv)
            self.canvas.plot_clf_map(hsv2rgb(db_map_new))
        elif ui_map == self.DMAPND2:
            db_map_new = transform_db_map(self.clfData.dist_nd2,
                                          self.clfData.dmap_hsv)
            self.canvas.plot_clf_map(hsv2rgb(db_map_new))
        elif ui_map == self.DMAPND3:
            db_map_new = transform_db_map(self.clfData.dist_nd3,
                                          self.clfData.dmap_hsv)
            self.canvas.plot_clf_map(hsv2rgb(db_map_new))

    def setHLNeighbors(self, checked):
        if self.highlightNeighbors is True:
            self.canvas.draw_lines_to_pts([0, 0], [])
        self.highlightNeighbors = checked

    def showPts(self, show_mode):
        self.show_pts_sel = show_mode
        if self.show_pts_sel == 0:  # show all pts
            colors = [COLORS[i] for i in self.clfData.y_pred]
            self.canvas.plot_proj(self.clfData.proj, colors)
        elif self.show_pts_sel == 1:  # show misclf pts
            colors = np.array([COLORS[i] for i in self.clfData.y_pred])
            miss_colors = colors[self.clfData.miss_idx]
            self.canvas.plot_proj(self.clfData.miss_proj, miss_colors)
        elif self.show_pts_sel == 2:  # show no pts (only boundary maps)
            self.canvas.plot_proj(None, None)

    def showAllPts(self):
        print('Show all pts')
        if self.show_pts_sel == 0:
            print('already selected, skipping')
            return
        self.show_pts_sel = 0
        colors = [COLORS[i] for i in self.clfData.y_pred]
        self.canvas.plot_proj(self.clfData.proj, colors)

    def showMissPts(self):
        print('Show miss pts')
        if self.show_pts_sel == 1:
            print('already selected, skipping')
            return
        self.show_pts_sel = 1
        colors = np.array([COLORS[i] for i in self.clfData.y_pred])
        miss_colors = colors[self.clfData.miss_idx]
        self.canvas.plot_proj(self.clfData.miss_proj, miss_colors)

    # TODO: make canvas not show a scatter plot
    def showNonePts(self):
        print('Show none pts')
        if self.show_pts_sel == 2:
            print('already selected, skipping')
            return
        self.show_pts_sel = 2

    # rec value of -1 means unselected
    def selectionChange(self, rec):
        # TODO: check showPts value
        # showPts == 0: point index is rec, change scatter_handler to show only
        #               selected point and neighbors
        # showPts == 1: point index is clfData.miss_idx[rec], change
        #               scatter_handler to show selected point and nD neighbors
        # showPts == 2: should never happen
        # print('selection change')
        # print('received: ', rec)

        # if self.showPts == 0:
        #     idx = rec
        # elif self.showPts == 1:
        #     idx = self.clfData.miss_idx[rec]

        # idx = self.getTrueIndex(rec)
        idx = rec
        true_class = self.clfData.y[idx]
        pred_class = self.clfData.y_pred[idx]

        n_info = self.clfData.getNeighborsInfo(idx)

        img = self.clfData.X[idx]*255
        img = img.reshape((IMG_SIZE, IMG_SIZE)).astype(np.uint8)

        nimgs = self.clfData.X[n_info['neighbors']]*255.0
        nimgs = nimgs.reshape((NUM_NEIGHBORS, IMG_SIZE, IMG_SIZE)).astype(np.uint8)

        self.errUI.set_point(idx, true_class, pred_class, n_info, img, nimgs)

    def on_press(self, event):
        print('on_press')
        if event.button == 1:
            return
        self.canvas.cur_xlim = self.canvas.axes.get_xlim()
        self.canvas.cur_ylim = self.canvas.axes.get_ylim()
        self.canvas.xpress, self.canvas.ypress = event.xdata, event.ydata
        self.canvas.press = True

    def on_release(self, event):
        print('on_release')
        self.canvas.press = False
        self.canvas.axes.figure.canvas.draw()

    def on_motion(self, event):
        if event.inaxes != self.canvas.axes:
            return

        if self.canvas.press is False:
            if self.freeze is True:
                return

            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return


            if self.show_pts_sel == 2:
                return

            # get the 2D point closest to the mouse
            closest_pt = self.clfData.getNNProj(x, y, self.show_pts_sel == 1)
            closest_pt = self.getTrueIndex(closest_pt)

            # get the 2D coords of the closest point
            closest_pt_coords = self.clfData.proj[closest_pt]
            # if we are highlighting neighbors, we draw lines to them
            if self.highlightNeighbors is True:
                # get neighborhood information of this point in nd
                neighbors_nd = self.clfData.getNeighborsInfo(closest_pt)
                # the the 2D coords of the nD neighbors
                neighbors_2d_coords = self.clfData.proj[neighbors_nd['neighbors']]
                self.canvas.draw_lines_to_pts(closest_pt_coords, neighbors_2d_coords)

            if self.show_closest_label is True:
                coords = np.array([[x/GRID_SIZE, y/GRID_SIZE]])
                sample = self.clfData.inv_proj.transform(coords, normalize=True)[0]

                closest_nd_idx, _ = self.clfData.get_neighbors_nD(sample)
                closest_nd_idx = closest_nd_idx[1]

                # check the label the classifier assigns to it
                label = self.clfData.y_pred[closest_nd_idx]
                label = SYN_HUES[label]

                # FIXME inverted?
                row = int(y)
                col = int(x)
                cell_label = self.clfData.dmap_hsv[row, col, 0]
                # cell_label = np.where(SYN_HUES == cell_label)

                if label != cell_label:
                    print("\t\tDIFFERENT LABELS: ", cell_label, label)
                else:
                    print("\t\tEQUAL LABELS: ", cell_label, label)

            if self.features is True:
                return

            # generates ilamp proposals in a small area around mouse click
            self.gen_ilamp_samples(x/GRID_SIZE, y/GRID_SIZE)

            # updates the images do the nD neighbors, distances and true/pred
            # classes
            self.selectionChange(closest_pt)

            # self.gen_random_samples(self.clfData.X[closest_pt])
            # self.gen_interp_samples(closest_pt)
            # row_db, col_db = self.closest_decision_boundary(x, y)
            # self.canvas.draw_lines_to_pts(closest_pt_coords, [[col_db + 0.5, row_db + 0.5]])
            return

        dx = event.xdata - self.canvas.xpress
        dy = event.ydata - self.canvas.ypress
        self.canvas.cur_xlim -= dx
        self.canvas.cur_ylim -= dy
        self.canvas.axes.set_xlim(self.canvas.cur_xlim)
        self.canvas.axes.set_ylim(self.canvas.cur_ylim)

        self.canvas.axes.figure.canvas.draw()
        self.canvas.axes.figure.canvas.flush_events()

    def on_scroll(self, event):
        print('on_scroll: ', vars(event))
        step = event.step
        base_scale = 1.0 + abs(step)*0.1

        cur_xlim = self.canvas.axes.get_xlim()
        cur_ylim = self.canvas.axes.get_ylim()

        if event.button == 'down':
            scale_factor = 1.0/base_scale
        else:  # event == 'up'
            scale_factor = base_scale

        xdata, ydata = event.xdata, event.ydata
        self.canvas.axes.set_xlim([xdata - (xdata-cur_xlim[0])/scale_factor, xdata + (cur_xlim[1]-xdata)/scale_factor]) 
        self.canvas.axes.set_ylim([ydata - (ydata-cur_ylim[0])/scale_factor, ydata + (cur_ylim[1]-ydata)/scale_factor])
        self.canvas.draw() # force re-draw

    def on_key_release(self, event):
        print(event.key)
        if event.key == 'b':
            x = event.xdata
            y = event.ydata

            print('looking for decision boundary')

            row_db, col_db = self.closest_decision_boundary(x, y)
            self.canvas.draw_lines_to_pts([x, y], [[col_db + 0.5, row_db + 0.5]])
            # coords = np.array([col_db + 0.5, row_db + 0.5])/GRID_SIZE
            # sample = lamp.ilamp_norm(self.clfData.X, self.clfData.proj_norm, coords)
            # print('2D distance: ', np.linalg.norm(np.array([x - col_db + 0.5, y - row_db + 0.5])))
            # diff_cell_db = self.inv_proj_samples[0].reshape((784,)) - sample.reshape((784,))
            # print('nD distance: ', np.linalg.norm(diff_cell_db))

    # def restoreSelectedPt(self):
    #     face_color = self.canvas.scatter_handler.get_facecolor()
    #     face_color[self.selected_pt] = self.prev_color
    #     self.errUI.clearSelection()
    #     self.selected_pt = -1

    # Auxiliary function that checks program state to find out if we are
    # dealing with index from misclf scatter or an index every point in proj
    def getTrueIndex(self, idx):
        if self.show_pts_sel == 0:
            return idx
        elif self.show_pts_sel == 1:
            return self.clfData.miss_idx[idx]

    # TODO precompute colors?
    def getColor(self, idx):
        return COLORS[self.clfData.y_pred[idx]]

    def gen_ilamp_samples(self, x, y):
        # SIZE = 1.0
        # cell_size = SIZE/GRID_SIZE
        # grid_pts = []
        # grid_pts.append((x, y))
        # limits = [x - cell_size*0.5, y - cell_size*0.5,
        #           x + cell_size*0.5, y + cell_size*0.5]
        # grid_pts.extend(SampleSquare(NUM_SAMPLES - 1, limits))

        limits = [x - self.ilamp_radius, y - self.ilamp_radius,
                  x + self.ilamp_radius, y + self.ilamp_radius]
        sample_size = int(np.sqrt(NUM_SAMPLES))
        grid_pts = SampleSquareUniform(sample_size, limits)
        for i in range(NUM_SAMPLES):
            coords = np.array([grid_pts[i]])
            # sample = lamp.ilamp_norm(self.clfData.X, self.clfData.proj_norm,
            #                          coords, k=NUM_NEIGHBORS)
            sample = self.clfData.inv_proj.transform(np.array(coords), normalize=True)[0]
            self.inv_proj_samples[i] = np.copy(sample)
            sample = (255.0*sample).reshape((IMG_SIZE, IMG_SIZE)).astype(np.uint8)
            qimage = QImage(sample, sample.shape[0], sample.shape[1],
                            QImage.Format_Grayscale8)
            self.newsamplesUI.syntheticImgs[i].setPixmap(QPixmap(qimage))

    def gen_random_samples(self, x):
        random_disturb = np.random.randn(NUM_SAMPLES, 784)
        norm = np.linalg.norm(random_disturb, axis=1)
        # normalizes the disturb vector, should set ball size
        for i in range(random_disturb.shape[0]):
            random_disturb[i] /= norm[i]
            random_disturb[i] *= 2

        new_samples = x + random_disturb

        for i in range(NUM_SAMPLES):
            vmin = new_samples[i].min()
            vmax = new_samples[i].max()
            new_sample = (new_samples[i] - vmin)/(vmax - vmin)

            new_sample = (255.0*new_sample).reshape((IMG_SIZE, IMG_SIZE)).astype(np.uint8)
            qimage = QImage(new_sample, new_sample.shape[0],
                            new_sample.shape[1], QImage.Format_Grayscale8)
            self.newsamplesUI.rndNoiseImgs[i].setPixmap(QPixmap(qimage))

    def gen_interp_samples(self, idx):
        x = self.clfData.X[idx]
        random_disturb = np.random.randn(NUM_SAMPLES, 784)
        norm = np.linalg.norm(random_disturb, axis=1)
        # normalizes the disturb vector, should set ball size
        for i in range(random_disturb.shape[0]):
            random_disturb[i] /= norm[i]
            random_disturb[i] *= self.nd_radius

        new_pts = x + random_disturb

        for i in range(NUM_SAMPLES):
            # get the nearest neighbor to new_pts[i] (!= x) and its distance d
            new_pt = new_pts[i]
            indices, distances = self.clfData.get_neighbors_nD(new_pt)
            # FIXME: check if this is correct
            if indices[0] == idx:
                nn = self.clfData.X[indices[1]]
                dist = distances[1]
            else:
                nn = self.clfData.X[indices[0]]
                dist = distances[0]
            # print("gen_interp_samples: ", self.nd_radius, dist)
            t = self.nd_radius/(self.nd_radius + dist)
            new_sample = Lerp(x, nn, t)
            # interpolate between x and new_pts[i] weighting the distances
            sample_min = new_sample.min()
            sample_max = new_sample.max()
            new_sample = (new_sample - sample_min)/(sample_max - sample_min)

            new_sample = (255.0*new_sample).reshape((IMG_SIZE, IMG_SIZE)).astype(np.uint8)
            qimage = QImage(new_sample, new_sample.shape[0], new_sample.shape[1],
                            QImage.Format_Grayscale8)
            self.newsamplesUI.rndNoiseImgs[i].setPixmap(QPixmap(qimage))

    def cell_neighbors(self, row, col):
        neighbors = []
        if row - 1 >= 0:
            neighbors.append([row - 1, col])
        if col - 1 >= 0:
            neighbors.append([row, col - 1])
        if row + 1 < GRID_SIZE:
            neighbors.append([row + 1, col])
        if col + 1 < GRID_SIZE:
            neighbors.append([row, col + 1])

        return neighbors

    def closest_decision_boundary(self, x, y):
        row = int(y)
        col = int(x)

        cell_label = self.clfData.dmap_hsv[row, col][0]
        cell_label2 = 0
        if cell_label in CMAP_ORIG:
            for i in range(len(CMAP_ORIG)):
                if cell_label == CMAP_ORIG[i]:
                    cell_idx = i
                    break
            cell_label2 = CMAP_SYN[cell_idx]
        else:
            for i in range(len(CMAP_SYN)):
                if cell_label == CMAP_SYN[i]:
                    cell_idx = i
                    break
            cell_label2 = CMAP_ORIG[cell_idx]

        queue = self.cell_neighbors(row, col)
        visited = [[row, col]]

        print('looking for nearest db: ', row, col)
        print('cell_label: ', cell_label)
        print('queue: ', queue)

        while queue:
            # row and col of the cell in the queue
            r, c = queue.pop(0)
            # label of this cell
            l = self.clfData.dmap_hsv[r, c][0]
            # print('iter ', curr_iter, 'label: ', l)

            if cell_label != l and cell_label2 != l:
                print('found!')
                return [r, c]

            visited.append([r, c])

            neighbors = self.cell_neighbors(r, c)
            for n in neighbors:
                if n in visited:
                    continue
                queue.append(n)

            # print('queue: ', queue)
            # print('visited: ', visited)
        return [0, 0]

    def openProjection(self):
        fname = QFileDialog.getOpenFileName(self, 'Open Projection File', '.')

        if fname[0]:
            self.features = self.clfData.FromJSON(fname[0])

            colors = [COLORS[i] for i in self.clfData.y_pred]
            self.canvas.set_scene(self.clfData.dmap, self.clfData.proj, colors)

            self.connect_evts()

    def save_new_samples(self):
        np.save('new_samples.npy', np.array(self.new_samples))
        np.save('new_samples_y.npy', np.array(self.new_samples_y))

    # def closeEvent(self, event):
    #     reply = QMessageBox.question(self, 'msg', 'quit?',
    #                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    #     if reply == QMessageBox.Yes:
    #         event.accept()
    #     else:
    #         event.ignore()


class CLFData():
    def __init__(self):
        self.X = None
        self.y = None
        self.proj = None
        self.dmap = None
        self.dist_map = None
        self.res = 0

        self.miss_proj = None
        self.miss_proj_nbrs = None

        self.inv_proj = None

    def FromJSON(self, filepath):
        with open(filepath) as f:
            data_json = json.load(f)

        grid = Grid()
        grid.load(data_json['grid'], data_json['clf'], data_json['inv_proj'])

        # self.inv_proj = grid.inv_proj
        # FIXME: guardar o tipo da projeção inversa (ilamp, rbf ou nn)
        inv_proj_type = data_json['inv_proj_type']
        if inv_proj_type == 'ilamp':
            self.inv_proj = ILAMP()
        elif inv_proj_type == 'rbf':
            self.inv_proj = RBFInv()
        elif inv_proj_type == 'nninv':
            self.inv_proj = NNInv()
        self.inv_proj.load(data_json['inv_proj'])

        self.X = np.copy(grid.X_nd)
        global IMG_SIZE
        IMG_SIZE = int(np.sqrt(self.X.shape[1]))

        self.y = np.load(data_json['y_true'])
        self.y_pred = np.load(data_json['y_pred'])

        self.proj = np.copy(grid.X_2d)

        self.dmap_hsv = grid.dmap
        # self.dmap = HSV2RGB(self.dmap_hsv)
        self.dmap = hsv2rgb(self.dmap_hsv)

        if grid.dist_2d is not None:
            self.dist_2d = grid.dist_2d
            self.dist_nd = grid.dist_nd
            self.dist_nd2 = grid.dist_nd2
            self.dist_nd3 = grid.dist_nd3

        global GRID_SIZE
        GRID_SIZE = grid.R
        print('grid_size: ', GRID_SIZE)

        self.proj_norm = np.copy(self.proj)
        self.proj *= GRID_SIZE
        self.nbrs = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1,
                                     algorithm='kd_tree')
        self.nbrs.fit(self.X)

        # nearest neighbors for the projected points to do the "brushing" thing
        self.proj_nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        self.proj_nbrs.fit(self.proj)

        # FIXME: make it work with 0 missclf
        self.miss_idx = np.arange(self.proj.shape[0], dtype=int)
        self.miss_idx = self.miss_idx[self.y != self.y_pred]
        if len(self.miss_idx) > 0:
            self.miss_proj = self.proj[self.miss_idx]
            self.miss_proj_nbrs = NearestNeighbors(n_neighbors=1,
                                                   algorithm='kd_tree')
            self.miss_proj_nbrs.fit(self.miss_proj)

        return "features" in data_json

    # miss == False means that we want the nearest neighbor considering all pts
    # miss == True means we want nearest neighbors only for misclf pts
    # FIXME: add a state to know which is being used
    # FIXME: move this function to the main window
    def getNNProj(self, x, y, miss=False):
        if miss is False:
            _, indices = self.proj_nbrs.kneighbors(np.array([[x, y]]))
        else:
            _, indices = self.miss_proj_nbrs.kneighbors(np.array([[x, y]]))
        return indices[0][0]

    def getNeighborsInfo(self, idx):
        if idx == -1:
            return None
        n_info = dict()
        distances, indices = self.nbrs.kneighbors([self.X[idx]])
        indices = indices[0][1:]
        distances = distances[0][1:]
        n_info['neighbors'] = indices
        n_info['distances'] = distances
        n_info['pred_label'] = self.y_pred[indices]
        n_info['true_label'] = self.y[indices]
        return n_info

    def get_neighbors_nD(self, x):
        distances, indices = self.nbrs.kneighbors([x])
        indices = indices[0]
        distances = distances[0]

        return indices, distances


class ErrUI(QWidget):
    def __init__(self):
        super().__init__()
        idxTxt = QLabel('Point:')
        self.idxLbl = QLabel('')
        trueTxt = QLabel('True Class:')
        self.trueLbl = QLabel('')
        predTxt = QLabel('Pred Class:')
        self.predLbl = QLabel('')

        # self.neighbors_list = QListWidget(self)
        self.neighbors_list = QTableWidget(self)
        self.neighbors_list.setRowCount(NUM_NEIGHBORS)

        # Columns: neighbors properties to check: id, pred, true, dist
        self.neighbors_list.setColumnCount(4)

        # inserting columns so it is possible to set their headers
        headerLabels = ["idx", "pred", "true", "distance"]
        self.neighbors_list.setHorizontalHeaderLabels(headerLabels)
        self.neighbors_list.setEditTriggers(QTableWidget.NoEditTriggers)

        neighborsLbl = QLabel('Nearest Neighbors:')

        ptImgTxt = QLabel('Img:')
        self.ptImg = QLabel(self)
        img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)*255
        qimage = QImage(img, img.shape[0], img.shape[1],
                        QImage.Format_Grayscale8)
        self.ptImg.setPixmap(QPixmap(qimage))

        nbImgTxt = QLabel('Neighbors Imgs:')
        self.neighborsImgs = [QLabel(self) for _ in range(NUM_NEIGHBORS)]
        for i in range(NUM_NEIGHBORS):
            self.neighborsImgs[i].setPixmap(QPixmap(qimage))

        inv_sample_lbl = QLabel('Inverse Projected Sample:')
        self.inv_sample = QLabel(self)
        img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)*255
        qimage = QImage(img, img.shape[0], img.shape[1],
                        QImage.Format_Grayscale8)
        self.inv_sample.setPixmap(QPixmap(qimage))


        hbox0 = QHBoxLayout()
        hbox0.addWidget(idxTxt)
        hbox0.addWidget(self.idxLbl)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(trueTxt)
        hbox1.addWidget(self.trueLbl)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(predTxt)
        hbox2.addWidget(self.predLbl)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(ptImgTxt)
        hbox3.addWidget(self.ptImg)

        hbox4 = QHBoxLayout()
        hbox4.addWidget(nbImgTxt)
        for i in range(NUM_NEIGHBORS):
            hbox4.addWidget(self.neighborsImgs[i])

        vbox = QVBoxLayout()
        vbox.addLayout(hbox0)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)

        # separator-like
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        vbox.addWidget(line)
        vbox.addWidget(neighborsLbl)
        vbox.addWidget(self.neighbors_list)

        line = QFrame(self)
        # line.setObjectName("");
        # line.setGeometry(QRect(320, 150, 118, 3));
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        vbox.addWidget(line)

        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)

        vbox.addStretch(1)

        self.setLayout(vbox)

    def clearSelection(self):
        self.idxLbl.setText('')
        self.trueLbl.setText('')
        self.predLbl.setText('')
        self.neighbors_list.clearContents()

        img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)*255
        qimage = QImage(img, img.shape[0], img.shape[1],
                        QImage.Format_Grayscale8)
        self.ptImg.setPixmap(QPixmap(qimage))
        for i in range(NUM_NEIGHBORS):
            self.neighborsImgs[i].setPixmap(QPixmap(qimage))

    def set_point(self, idx, true, pred, n_info, img, neighbors_imgs):
        self.neighbors_list.clearContents()
        if idx == -1:
            self.clearSelection()
            return

        self.idxLbl.setText(str(idx))
        self.trueLbl.setText(str(true))
        self.predLbl.setText(str(pred))

        qimg = QImage(img, img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        self.ptImg.setPixmap(QPixmap(qimg))

        for i in range(NUM_NEIGHBORS):
            img = neighbors_imgs[i]
            qimg = QImage(img, img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            self.neighborsImgs[i].setPixmap(QPixmap(qimg))

        neighbors_indices = n_info['neighbors']
        for i in range(len(neighbors_indices)):
            n = neighbors_indices[i]
            item = QTableWidgetItem(str(n))
            self.neighbors_list.setItem(i, 0, item)

        neighbors_pred = n_info['pred_label']
        for i in range(len(neighbors_pred)):
            n = neighbors_pred[i]
            item = QTableWidgetItem(str(n))
            self.neighbors_list.setItem(i, 1, item)

        neighbors_true = n_info['true_label']
        for i in range(len(neighbors_true)):
            n = neighbors_true[i]
            item = QTableWidgetItem(str(n))
            self.neighbors_list.setItem(i, 2, item)

        neighbors_distances = n_info['distances']
        for i in range(len(neighbors_distances)):
            n = neighbors_distances[i]
            item = QTableWidgetItem(str(n))
            self.neighbors_list.setItem(i, 3, item)


class NewSamplesUI(QWidget):
    def __init__(self):
        super().__init__()
        nsTxt = QLabel('Inverse Projected samples:')
        img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)*255
        qimage = QImage(img, img.shape[0], img.shape[1],
                        QImage.Format_Grayscale8)
        self.syntheticImgs = [QLabel(self) for _ in range(NUM_SAMPLES)]
        for i in range(NUM_SAMPLES):
            self.syntheticImgs[i].setPixmap(QPixmap(qimage))

        # nsNoiseTxt = QLabel('nD samples:')
        # self.rndNoiseImgs = [QLabel(self) for _ in range(NUM_SAMPLES)]
        # for i in range(NUM_SAMPLES):
        #     self.rndNoiseImgs[i].setPixmap(QPixmap(qimage))

        syn_grid = QGridLayout()
        size = np.sqrt(NUM_SAMPLES).astype(np.int)
        for i in range(size):
            for j in range(size):
                idx = size*i + j
                syn_grid.addWidget(self.syntheticImgs[idx], i, j)

        # nD_grid = QGridLayout()
        # for i in range(size):
        #     for j in range(size):
        #         idx = size*i + j
        #         nD_grid.addWidget(self.rndNoiseImgs[idx], i, j)

        row_spin = QSpinBox(self)
        row_spin.setRange(0, 4)
        col_spin = QSpinBox(self)
        col_spin.setRange(0, 4)

        save_btn = QPushButton("Save", self)

        # grid = QGridLayout()
        # # imgs on row 0
        # for i in range(NUM_SAMPLES):
        #     grid.addWidget(self.syntheticImgs[i], 0, i)
        # for i in range(NUM_SAMPLES):
        #     grid.addWidget(self.ilamp_labels[i], 1, i)
        # for i in range(NUM_SAMPLES):
        #     grid.addWidget(self.add_sample_btn[i], 2, i)

        # hbox1 = QHBoxLayout()
        # for ni in self.rndNoiseImgs:
        #     hbox1.addWidget(ni)

        save_hbox = QHBoxLayout()
        save_hbox.addWidget(QLabel("Row"))
        save_hbox.addWidget(row_spin)
        save_hbox.addWidget(QLabel("Col"))
        save_hbox.addWidget(col_spin)
        save_hbox.addWidget(save_btn)

        vbox0 = QVBoxLayout()
        vbox0.addWidget(nsTxt)
        # vbox0.addLayout(grid)
        vbox0.addLayout(syn_grid)
        vbox0.addLayout(save_hbox)
        # vbox0.addLayout(hbox0)
        # vbox0.addLayout(hbox2)
        # vbox0.addWidget(nsNoiseTxt)
        # vbox0.addLayout(nD_grid)
        # vbox0.addLayout(hbox0)
        # vbox0.addLayout(hbox1)
        vbox0.addStretch(1)
        self.setLayout(vbox0)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

