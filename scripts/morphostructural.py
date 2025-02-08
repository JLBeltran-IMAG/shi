import numpy as np
import numpy.ma as ma

import pandas as pd

from scipy.stats import norm
from scipy.spatial import distance as distance_measure

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Ellipse, Polygon
from matplotlib.text import Text
from matplotlib.path import Path as path_draw

from matplotlib.backend_tools import ToolBase, ToolToggleBase
plt.rcParams['toolbar'] = 'toolmanager'

from matplotlib.widgets import Button, RadioButtons, EllipseSelector, RectangleSelector, PolygonSelector, RangeSlider
from PySide6 import QtWidgets
from PySide6.QtGui import QIcon

import skimage.io as io
from skimage.exposure import rescale_intensity

from itertools import product
from pathlib import Path
from datetime import datetime
import argparse
import sys

import scripts.correlation as corr_ellipse


default_color = list(mcolors.TABLEAU_COLORS.values())



class SimpleLegendLabelSaver(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.saved_label = ""
        self.saved_color = None
        self.init_ui()

    def init_ui(self):
        self.layout = QtWidgets.QHBoxLayout()

        self.text_box = QtWidgets.QLineEdit(self)
        self.text_box.setPlaceholderText("class ...")
        self.layout.addWidget(self.text_box)

        self.button = QtWidgets.QPushButton("Add", self)
        self.button.clicked.connect(self.save_text)
        self.layout.addWidget(self.button)

        self.button_select_color = QtWidgets.QPushButton("Color", self)
        self.button_select_color.clicked.connect(self.select_color)
        self.layout.addWidget(self.button_select_color)

        self.setLayout(self.layout)
        self.setWindowTitle("Adding label for legend")
        self.resize(350, 50)

    def save_text(self):
        text = self.text_box.text()
        if text:
            self.saved_label = text
            QtWidgets.QMessageBox.information(self, "Saved", "Label saved correctly")
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "Advertencia", "El cuadro de texto está vacío")

    def select_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.saved_color = color.getRgbF()
        else:
            self.saved_color = "red"





class ImageContainer2D:
    def __init__(self, ax, image):
        self.image = rescale_intensity(image, out_range = (0, 255))

        self.ax = ax
        self.ax.axis("off")

        self.image_display = self.ax.imshow(self.image, cmap = 'gray')

        self.divider_slide = make_axes_locatable(self.ax)
        self.ax_slider = self.divider_slide.append_axes("bottom", size = "3%", pad = "3%")
        self.slider = RangeSlider(self.ax_slider, label = "", valmin = self.image.min(), valmax = self.image.max())
        self.slider.on_changed(self.change_values_slider)


    def change_values_slider(self, val):
        self.image_display.norm.vmin = val[0]
        self.image_display.norm.vmax = val[1]
        self.ax.get_figure().canvas.draw_idle()
        self.ax.get_figure().show()


class PixelWise(ImageContainer2D):
    def __init__(self, ax, image, type_of_contrast, class_input_mode = False):
        super().__init__(ax, image)
        self.ax = ax

        self.type_of_contrast = type_of_contrast
        self.obj_creation = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data = dict()

        # Distance Wasser bla bla bla
        self.u_values = list()
        self.v_values = list()

        # self.data4clusters = pd.DataFrame(columns = ["abs", "scatt", "class"])

        self.labels = 0
        self._colors = list()
        self.class_input_mode = class_input_mode

        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.verts = None

        self.mask = None
        self.patch = None

        self.rect = RectangleSelector(
            ax,
            self.rectangle_callback,
            useblit = True,
            button = [1, 3],
            minspanx = 5,
            minspany = 5,
            spancoords = 'pixels',
            interactive = True
            )

        self.ellipse = EllipseSelector(
            ax,
            self.ellipse_callback,
            useblit = True,
            button = [1, 3],
            minspanx = 5,
            minspany = 5,
            spancoords = 'pixels',
            interactive = True
            )
        self.ellipse.set_active(False)

        self.polygon = PolygonSelector(ax, self.polygon_callback, useblit = True)
        self.polygon.set_active(False)

        self.fig = None

        # self.activate_deactivate_selector = fig.canvas.mpl_connect('key_press_event', self.toggle_selector)
        fig.canvas.mpl_connect('key_press_event', self.set_unset_class)


    def polygon_callback(self, verts):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        if self.type_of_contrast == "abs":
            self.verts = verts
            two.verts = self.verts

            self.patch = Polygon(self.verts, facecolor = 'none')
            two.patch = Polygon(self.verts, facecolor = 'none')

        else:
            self.verts = verts
            one.verts = self.verts

            self.patch = Polygon(self.verts, facecolor = 'none')
            one.patch = Polygon(self.verts, facecolor = 'none')


    def rectangle_callback(self, eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        self.x0, self.y0 = round(eclick.xdata), round(eclick.ydata)
        self.x1, self.y1 = round(erelease.xdata), round(erelease.ydata)

        self.y0 = min(self.y0, self.y1)
        self.y1 = max(self.y0, self.y1)
        self.x0 = min(self.x0, self.x1)
        self.x1 = max(self.x0, self.x1)

        if self.type_of_contrast == "abs":
            two.y0 = self.y0
            two.y1 = self.y1
            two.x0 = self.x0
            two.x1 = self.x1

            self.patch = Polygon(np.column_stack(self.rect.corners), facecolor = 'none')
            two.patch = Polygon(np.column_stack(self.rect.corners), facecolor = 'none')

        else:
            one.y0 = self.y0
            one.y1 = self.y1
            one.x0 = self.x0
            one.x1 = self.x1

            self.patch = Polygon(np.column_stack(self.rect.corners), facecolor = 'none')
            one.patch = Polygon(np.column_stack(self.rect.corners), facecolor = 'none')


    def ellipse_callback(self, eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        self.x0, self.y0 = round(eclick.xdata), round(eclick.ydata)
        self.x1, self.y1 = round(erelease.xdata), round(erelease.ydata)

        self.y0 = min(self.y0, self.y1)
        self.y1 = max(self.y0, self.y1)
        self.x0 = min(self.x0, self.x1)
        self.x1 = max(self.x0, self.x1)

        if self.type_of_contrast == "abs":
            two.y0 = self.y0
            two.y1 = self.y1
            two.x0 = self.x0
            two.x1 = self.x1

            self.patch = Ellipse(self.ellipse.center, width = abs(self.x1 - self.x0), height = abs(self.y1 - self.y0), facecolor = 'none')
            two.patch = Ellipse(self.ellipse.center, width = abs(self.x1 - self.x0), height = abs(self.y1 - self.y0), facecolor = 'none')

        else:
            one.y0 = self.y0
            one.y1 = self.y1
            one.x0 = self.x0
            one.x1 = self.x1

            self.patch = Ellipse(self.ellipse.center, width = abs(self.x1 - self.x0), height = abs(self.y1 - self.y0), facecolor = 'none')
            one.patch = Ellipse(self.ellipse.center, width = abs(self.x1 - self.x0), height = abs(self.y1 - self.y0), facecolor = 'none')


    def define_type_selector(self, label):
        if label == 'Sq.':
            self.rect.set_active(True)
            self.ellipse.set_active(False)
            self.polygon.set_active(False)

        elif label == 'Ell.':
            self.rect.set_active(False)
            self.ellipse.set_active(True)
            self.polygon.set_active(False)

        elif label == 'Poly.':
            self.rect.set_active(False)
            self.ellipse.set_active(False)
            self.polygon.set_active(True)

        else: pass


    def set_unset_class(self, event):
        if event.key == 'a':
            if self.class_input_mode:
                label_legend = SimpleLegendLabelSaver()
                label_legend.exec()
                exp_label = label_legend.saved_label
                exp_color = label_legend.saved_color
                self._colors.append(exp_color)

            else:
                exp_label = "class_{}".format(self.labels)
                exp_color = default_color[self.labels]
                self._colors.append(exp_color)

            if self.polygon.active:
                xmin, ymin = np.min(self.verts, axis = 0)
                self.ax.text(xmin - 2, ymin - 10, "{}".format(exp_label), color = exp_color, label = "yes")
                self.patch.set_edgecolor(exp_color)
                self.ax.add_patch(self.patch)
                self.ax.get_figure().canvas.draw()

                self.pixel_wise_absorption_scattering(exp_label)

                self.rect.clear()
                self.ellipse.clear()
                self.polygon.clear()

            elif self.x0 != self.x1 and self.y0 != self.y1:
                self.ax.text(self.x0 - 2, self.y0 - 10, "{}".format(exp_label), color = exp_color, label = "yes")
                self.patch.set_edgecolor(exp_color)
                self.ax.add_patch(self.patch)
                self.ax.get_figure().canvas.draw()

                self.pixel_wise_absorption_scattering(exp_label)

                self.rect.clear()
                self.ellipse.clear()
                self.polygon.clear()


    def export_pxwise(self, legend_text):
        log = ma.log(self.mask).compressed()

        path_to_pixelwise_data = Path("pixel_wise", "pxw_{}".format(self.obj_creation))
        path_to_pixelwise_data.mkdir(parents = True, exist_ok = True)

        path = path_to_pixelwise_data.joinpath("{}_{}.txt".format(legend_text, self.type_of_contrast))

        np.savetxt(path, log)

        self.data["{}".format(legend_text)] = list()
        self.labels += 1
        self.mask = None


    def pixel_wise_absorption_scattering(self, legend_text):
        self.mask = ma.array(self.image, mask = True)

        if self.rect.active:
            self.mask.mask[self.y0 : self.y1, self.x0 : self.x1] = False

            self.export_pxwise(legend_text)

        elif self.ellipse.active:
            x_0, y_0 = (self.x1 + self.x0) / 2, (self.y1 + self.y0) / 2
            semi_major = (self.x1 - self.x0) / 2
            semi_minor = (self.y1 - self.y0) / 2

            y, x = np.ogrid[:self.image.shape[0], :self.image.shape[1]]
            mask = (((x - x_0) / semi_major) ** 2 + ((y - y_0) / semi_minor) ** 2) <= 1
            self.mask.mask = np.logical_not(mask)

            self.export_pxwise(legend_text)


        elif self.polygon.active:
            path = path_draw(self.verts)
            y, x = np.mgrid[:self.image.shape[0], :self.image.shape[1]]
            points = np.vstack((x.ravel(), y.ravel())).T
            mask = path.contains_points(points).reshape(self.image.shape)

            self.mask.mask = np.logical_not(mask)

            self.export_pxwise(legend_text)


    def _createing_dict_pxw(self, projection3D):
        path_to_pixelwise_data = [data for data in Path("pixel_wise").iterdir() if data.is_dir() and self.obj_creation in data.name]

        if len(path_to_pixelwise_data) != 0:
            path_abs = [data for data in path_to_pixelwise_data[0].iterdir() if data.is_file() and "abs" in data.name]
            path_scatt = [data for data in path_to_pixelwise_data[0].iterdir() if data.is_file() and "scatt" in data.name]

            if projection3D:
                path_phase = [data for data in path_to_pixelwise_data[0].iterdir() if data.is_file() and "phase" in data.name]

                for key in self.data.keys():
                    p2absrp = [path_to for path_to in path_abs if key in path_to.name][0]
                    p2scatt = [path_to for path_to in path_scatt if key in path_to.name][0]
                    p2phase = [path_to for path_to in path_phase if key in path_to.name][0]
                    value_to_data_dict = [p2absrp, p2scatt, p2phase]
                    self.data[key] = value_to_data_dict
            else:
                for key in self.data.keys():
                    p2absrp = [path_to for path_to in path_abs if key in path_to.name][0]
                    p2scatt = [path_to for path_to in path_scatt if key in path_to.name][0]
                    value_to_data_dict = [p2absrp, p2scatt]
                    self.data[key] = value_to_data_dict


    def graph_pixel_wise_absorption_scattering(self, event, projection3D):
        if self.type_of_contrast == "abs": self._createing_dict_pxw(projection3D)
        elif self.type_of_contrast == "scatt": self._createing_dict_pxw(projection3D)
        else: pass

        fig_pxw = plt.figure(figsize = (16, 8), layout = 'tight')

        if len(self.data) != 0:
            if projection3D:
                pass
                # ax_pxw = fig_pxw.add_subplot(1, 1, 1, projection = '3d')

                # for keys, values in zip(self.data.keys(), self.data.values()):
                #     pass
                #     abs_data = np.loadtxt(values[0])
                #     scatt_data = np.loadtxt(values[1])
                #     phase_data = np.loadtxt(values[2])
                #     ax_pxw.scatter(abs_data, scatt_data, phase_data, label = keys, alpha = 0.4)
                #     ax_pxw.set_xlabel("log(absorption)")
                #     ax_pxw.set_ylabel("log(scattering)")
                #     ax_pxw.set_zlabel("log(phase)")
                #     ax_pxw.set_proj_type('persp', focal_length=0.2)
                #     ax_pxw.legend()
            else:
                # -------------------------------------------------------------------------------------------------------------------------
                global_gs = gridspec.GridSpec(1, 2, figure = fig_pxw)

                gs0 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec = global_gs[0], width_ratios = (4, 1), height_ratios = (1, 4), wspace = 0.05, hspace = 0.05)
                gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = global_gs[1], wspace = 0.2, hspace = 0.3)

                ax_pxw = fig_pxw.add_subplot(gs0[1, 0])
                ax_pxw_hist_abs = fig_pxw.add_subplot(gs0[0, 0], sharex = ax_pxw)
                ax_pxw_hist_scatt = fig_pxw.add_subplot(gs0[1, 1], sharey = ax_pxw)

                ax_pxw_hist_fit_abs = fig_pxw.add_subplot(gs1[0, 0])
                ax_pxw_hist_fit_scatt = fig_pxw.add_subplot(gs1[1, 0])

                for index, (keys, values) in enumerate(self.data.items()):
                    # -------------------------------------------------------------------------------------------------------------------------
                    abs_data = np.loadtxt(values[0])
                    scatt_data = np.loadtxt(values[1])

                    _color = self._colors[index]

                    # self.data4clusters = pd.concat([self.data4clusters, pd.DataFrame({"abs": abs_data, "scatt": scatt_data, "class": keys})])

                    # -------------------------------------------------------------------------------------------------------------------------
                    hist_abs, intervals_abs = np.histogram(abs_data, bins = 100, density = True)

                    rv_norm_abs = norm
                    rv_norm_abs.fit(hist_abs)

                    mean_abs = np.mean(abs_data)
                    std_abs = np.std(abs_data)

                    x_axis_abs = np.linspace(intervals_abs.min(), intervals_abs.max(), 100)
                    line_abs = ax_pxw_hist_fit_abs.stairs(hist_abs, intervals_abs, edgecolor = _color)
                    ax_pxw_hist_fit_abs.plot(x_axis_abs, rv_norm_abs.pdf(x_axis_abs, mean_abs, std_abs), c = _color)
                    ax_pxw_hist_fit_abs.set_title("Histogram and gaussian fit - absorption")
                    ax_pxw_hist_fit_abs.set_xlabel("log(absoprtion)")
                    ax_pxw_hist_fit_abs.set_ylabel("counts")
                    ax_pxw_hist_fit_abs.grid(True)

                    hist_scatt, intervals_scatt = np.histogram(scatt_data, bins = 100, density = True)

                    rv_norm_scatt = norm
                    rv_norm_scatt.fit(hist_abs)

                    mean_scatt = np.mean(scatt_data)
                    std_scatt = np.std(scatt_data)

                    x_axis_scatt = np.linspace(intervals_scatt.min(), intervals_scatt.max(), 100)
                    ax_pxw_hist_fit_scatt.stairs(hist_scatt, intervals_scatt, edgecolor = _color)
                    ax_pxw_hist_fit_scatt.plot(x_axis_scatt, rv_norm_scatt.pdf(x_axis_scatt, mean_scatt, std_scatt), c = _color)
                    ax_pxw_hist_fit_scatt.set_title("Histogram and gaussian fit - scattering")
                    ax_pxw_hist_fit_scatt.set_xlabel("log(scattering)")
                    ax_pxw_hist_fit_scatt.set_ylabel("counts")
                    ax_pxw_hist_fit_scatt.grid(True)

                    # -------------------------------------------------------------------------------------------------------------------------
                    self.u_values.append(hist_abs[0])

                    # -------------------------------------------------------------------------------------------------------------------------
                    ax_pxw_hist_abs.tick_params(axis = "x", labelbottom = False)
                    ax_pxw_hist_scatt.tick_params(axis = "y", labelleft = False)
                    ax_pxw_hist_abs.hist(abs_data, bins = 100, density = True, alpha = 0.5)
                    ax_pxw_hist_scatt.hist(scatt_data, bins = 100, orientation = 'horizontal', density = True, alpha = 0.5)

                    # -------------------------------------------------------------------------------------------------------------------------
                    ax_pxw.scatter(abs_data, scatt_data, label = keys, alpha = 0.4, c = _color)
                    corr_ellipse.confidence_ellipse(abs_data, scatt_data, ax_pxw, line_abs.get_edgecolor(), confidence_level = 0.99, degrees_of_freedom = 2)

                    ax_pxw.scatter(mean_abs, mean_scatt, c = "black")

                    ax_pxw.axvline(mean_abs, c = "black", linestyle = "dashed")
                    ax_pxw.axhline(mean_scatt, c = "black", linestyle = "dashed")

                    ax_pxw.set_xlabel("log(absorption)")
                    ax_pxw.set_ylabel("log(scattering)")
                    ax_pxw.legend()

                ax_pxw.set_xlim()
                ax_pxw.set_ylim()


            distances_matrix = list()

            # for distances_values_abs in product(self.u_values, self.u_values):
            #     distances_matrix.append(distance_measure.correlation(distances_values_abs[0], distances_values_abs[1]))

            # print(np.array(distances_matrix).reshape((len(self.u_values), len(self.u_values))))

        else:
            pass

        # data_sep = self.data4clusters.drop(columns = ["class"])
        # target_sep = self.data4clusters["class"]

        # number_of_clusters = pd.unique(self.data4clusters["class"]).size
        # # model = KMeans(n_clusters = number_of_clusters).fit(data_sep)
        # model = DecisionTreeClassifier().fit(data_sep, target_sep)
        # # model = KNeighborsClassifier(n_neighbors = number_of_clusters).fit(data_sep, target_sep)

        # disp = DecisionBoundaryDisplay.from_estimator(model, data_sep, response_method = "predict")
        # disp.plot(ax = ax_pxw, alpha = 0.3)

        ax_pxw.grid(True)
        fig_pxw.show()


def figure_classes_to_export(path_to_export, obj1, obj2):
    fig_to_export = plt.figure(figsize = (12, 6), layout = "tight")
    ax1_to_export = fig_to_export.add_subplot(121)
    ax2_to_export = fig_to_export.add_subplot(122)

    patches = obj1.ax.findobj(Polygon)
    if len(patches) != 0:
        for patch in patches:
            patch = Polygon(patch.xy, facecolor = 'none', edgecolor = patch.get_edgecolor())
            ax1_to_export.add_patch(patch)

    patches = obj1.ax.findobj(Ellipse)
    if len(patches) != 0:
        for patch in patches:
            patch = Ellipse(patch.center, patch.width, patch.height, facecolor = 'none', edgecolor = patch.get_edgecolor())
            ax1_to_export.add_patch(patch)

    patches = obj2.ax.findobj(Polygon)
    if len(patches) != 0:
        for patch in patches:
            patch = Polygon(patch.xy, facecolor = 'none', edgecolor = patch.get_edgecolor())
            ax2_to_export.add_patch(patch)

    patches = obj2.ax.findobj(Ellipse)
    if len(patches) != 0:
        for patch in patches:
            patch = Ellipse(patch.center, patch.width, patch.height, facecolor = 'none', edgecolor = patch.get_edgecolor())
            ax2_to_export.add_patch(patch)


    texts = [txt for txt in obj1.ax.findobj(Text) if txt.get_label() == "yes"]
    if len(texts) != 0:
        for text in texts:
            x, y = text.get_position()
            patch = Text(x, y, text = text.get_text(), fontproperties = text.get_fontproperties(), color = text.get_color())
            ax1_to_export.add_artist(patch)

    texts = [txt for txt in obj2.ax.findobj(Text) if txt.get_label() == "yes"]
    if len(texts) != 0:
        for text in texts:
            x, y = text.get_position()
            patch = Text(x, y, text = text.get_text(), fontproperties = text.get_fontproperties(), color = text.get_color())
            ax2_to_export.add_artist(patch)


    img_ax1 = ax1_to_export.imshow(rescale_intensity(obj1.image, out_range=(0, 255)), vmin = obj1.slider.val[0], vmax = obj1.slider.val[1], cmap = "gray")
    divider_colorbar1 = make_axes_locatable(ax1_to_export)
    ax_colorbar1 = divider_colorbar1.append_axes("right", size = "2%", pad = "2%")
    fig_to_export.colorbar(img_ax1, ax_colorbar1, orientation = "vertical")

    img_ax2 = ax2_to_export.imshow(rescale_intensity(obj2.image, out_range=(0, 255)), vmin = obj2.slider.val[0], vmax = obj2.slider.val[1], cmap = "gray")
    divider_colorbar2 = make_axes_locatable(ax2_to_export)
    ax_colorbar2 = divider_colorbar2.append_axes("right", size = "2%", pad = "2%")
    fig_to_export.colorbar(img_ax2, ax_colorbar2, orientation = "vertical")

    ax1_to_export.axis("off")
    ax2_to_export.axis("off")

    fig_to_export.savefig(path_to_export)


def export_pxw_results(event, obj1, obj2):
    default_folder = Path().home().as_posix()
    default_filename = "morphostructural.png"

    msgBox = QtWidgets.QMessageBox()
    msgBox.setWindowTitle("Confirmation")
    msgBox.setText("Do you want to save your results?")
    msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
    msgBox.setDefaultButton(QtWidgets.QMessageBox.Ok)

    returnValue = msgBox.exec()

    if returnValue == QtWidgets.QMessageBox.Ok:
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setWindowTitle("Choose a filename to save to")
        file_dialog.setDirectory(default_folder)
        file_dialog.selectFile(default_filename)
        file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Portable Network Graphics Files (*.png)")

        file_dialog.exec()

        figure_classes_to_export(file_dialog.selectedFiles()[0], obj1, obj2)

    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Tool script - python script")
    parser.add_argument("--manually", action = "store_true", help = "correct angle or not")
    parser.add_argument("--select_folder", action = "store_true", help = "select folder")
    parser.add_argument("--input",  help = "correct angle or not")
    args = parser.parse_args()

    if args.select_folder:
        app = QtWidgets.QApplication([])
        default_folder = Path().home().joinpath("Documents/CXI/CXI-DATA-ANALYSIS").as_posix()
        file_abs, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Choose folder for ABSORPTION-CONTRAST", default_folder, "TIF (*.tif)")
        file_scatt, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Choose folder SCATTERING-CONTRAST", default_folder, "TIF (*.tif)")
        app.quit()
    
    else:
        pass

    if len(file_abs) != 0:
        imagexy1 = io.imread(file_abs)
        imagexy2 = io.imread(file_scatt)

        fig = plt.figure(figsize = (18, 9))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        one = PixelWise(ax1, imagexy1, "abs", args.manually)
        two = PixelWise(ax2, imagexy2, "scatt", args.manually)

        # --------------------------------------------------------------------------------------------------------------------------------
        def define_selector(label):
            fig.canvas.draw()
            one.define_type_selector(label)
            two.define_type_selector(label)

        ax_radio_buttons = fig.add_axes([0.94, 0.84, 0.05, 0.15], label = "button2")
        radio_buttons = RadioButtons(ax = ax_radio_buttons, labels = ["Sq.", "Ell.", "Poly."])
        radio_buttons.on_clicked(define_selector)
        # --------------------------------------------------------------------------------------------------------------------------------

        class CustomSave(ToolBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.image = '/home/beltran/Documents/cxi/icons/save.svg'

            def trigger(self, sender, event, data = None):
                export_pxw_results(event, one, two)


        class Morphostructural(ToolBase):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.image = '/home/beltran/Documents/cxi/icons/morpho.svg'

            def trigger(self, sender, event, data = None):
                one.graph_pixel_wise_absorption_scattering(event, False)


        fig.canvas.manager.toolmanager.add_tool('CustomSave', CustomSave)
        fig.canvas.manager.toolbar.add_tool('CustomSave', 'Saving custom figure')

        fig.canvas.manager.toolmanager.add_tool('Morphostructural', Morphostructural)
        fig.canvas.manager.toolbar.add_tool('Morphostructural', 'Morphostructural characterization')

        fig.canvas.manager.toolmanager.remove_tool("subplots")
        fig.canvas.manager.toolmanager.remove_tool("forward")
        fig.canvas.manager.toolmanager.remove_tool("back")
        fig.canvas.manager.toolmanager.remove_tool("help")
        fig.canvas.manager.toolmanager.remove_tool("save")

        plt.get_current_fig_manager().window.showMaximized()
        plt.get_current_fig_manager().window.setWindowTitle("Morphostructural analysis")
        plt.get_current_fig_manager().window.setWindowIcon(QIcon("/home/beltran/Documents/cxi/icons/win_morpho.svg"))
        plt.show()
    
    else:
        print("No file selected")
        sys.exit()


