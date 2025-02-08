import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.exposure import equalize_hist
from skimage.exposure import histogram


class cropImage:
    def __init__(self, ax, image, tmp_dir):
        self.image_rawdata = image
        self.image = equalize_hist(image)
        self.ax = ax
        self.ax.axis("off")

        image_display = self.ax.imshow(self.image, cmap = 'gray')

        divider = make_axes_locatable(self.ax)
        ax_colorbar = divider.append_axes("right", size = "3%", pad = "2%")
        self.ax.get_figure().colorbar(image_display, ax_colorbar, orientation = "vertical")

        self.tmp_dir = tmp_dir

    def setting_crop(self):
        xlim = self.ax.get_xbound()
        ylim = self.ax.get_ybound()
        limits = np.concatenate([ylim, xlim])
        np.savetxt("{}/crop.txt".format(self.tmp_dir), limits)

    def _calculating_snr(self, data):
        snr_1D_x = 10 * np.log10(np.max(data, axis = 0) / np.mean(data, axis = 0))
        snr_1D_y = 10 * np.log10(np.max(data, axis = 1) / np.mean(data, axis = 1))
        return snr_1D_x, snr_1D_y

    def exporting_image_inset(self):
        xlim = self.ax.get_xbound()
        ylim = self.ax.get_ybound()

        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]

        fig_inset, (ax_inset, ax_hist) = plt.subplots(1, 2)
        image_display = ax_inset.imshow(self.image, cmap = "gray")
        ax_inset.axis("off")

        divider = make_axes_locatable(ax_inset)
        ax_colorbar = divider.append_axes("right", size = "3%", pad = "2%")

        inset = inset_axes(ax_inset, width="40%", height="40%")
        inset.imshow(self.image[round(ylim[0]) : round(ylim[1]), round(xlim[0]) : round(xlim[1])], cmap = "gray")
        inset.axis("off")
        fig_inset.colorbar(image_display, ax_colorbar, orientation = "vertical")

        rectangle = plt.Rectangle((xlim[0], ylim[0]), width, height, edgecolor='red', facecolor = 'none')
        ax_inset.add_patch(rectangle)

        snr_1D_x, snr_1D_y = self._calculating_snr(self.image_rawdata[round(ylim[0]) : round(ylim[1]), round(xlim[0]) : round(xlim[1])])
        inset_histogram = histogram(self.image_rawdata[round(ylim[0]) : round(ylim[1]), round(xlim[0]) : round(xlim[1])])
        ax_hist.plot(inset_histogram[1], inset_histogram[0])
        ax_hist.set_title("Histogram of Region Of Interest (ROI)")
        ax_hist.set_xlabel("Gray values (arb. units)")
        ax_hist.set_ylabel("Count")

        divider = make_axes_locatable(ax_hist)
        ax_snr = divider.append_axes("bottom", size = "100%", pad = "25%")
        ax_snr.plot(snr_1D_x, label = "SNR along x-direction")
        ax_snr.plot(snr_1D_y, label = "SNR along y-direction")
        ax_snr.set_title("Signal-to-noise-ratio (SNR) of ROI")
        ax_snr.set_xlabel("Pixel place (arb. units)")
        ax_snr.set_ylabel("SNR (dB)")
        ax_snr.legend()

        fig_inset.show()

