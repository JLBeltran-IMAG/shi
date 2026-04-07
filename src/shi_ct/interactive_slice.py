import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage.io as io
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from pathlib import Path
import argparse



def slicing_3D_tif_tiff_file(filename_abs, filename_scat):
	# def read_tif_tiff():
	# 	iter_path_0 = Path('.').glob('*.tif')
	# 	iter_path_1 = Path('.').glob('*.tiff')

	# 	name_0 = [i for i in iter_path_0]
	# 	name_1 = [i for i in iter_path_0]

	# 	if len(name_0) == 1: and len(name_1) == 0
	# 		return name_0[0].name

	# 	elif len(name_0) == 0 and len(name_1) == 1:
	# 		return name_1[0].name

	# 	elif len(name_0) == 0 and len(name_1) == 0:
	# 		print("Files '*.tif' or '*.tiff' are missing")
	# 		return '0'

	# 	elif len(name_0) == 1: and len(name_1) == 1:
	# 		print("Too many files to process")
	# 		return '0'

	# 	else:
	# 		print("Too many files to process")
	# 		return '0'

	# image_file_to_open = read_tif_tiff()

	img_abs = io.imread(filename_abs)
	img_scat = io.imread(filename_scat)

	fig = plt.figure()

	initializer = 0
	display_abs = []
	display_scat = []

	graph_axes = []
	slider_axes = []
	button_axes = []
	save_axes = []

	display_graph = []
	display_slider = []
	display_button = []
	display_save = []

	graph = []
	slider = []
	button = []
	save = []

	# Creating axes
	graph_axes.append([0.00, 0.1, 0.28, 0.35])
	graph_axes.append([0.34, 0.1, 0.28, 0.35])
	graph_axes.append([0.67, 0.1, 0.28, 0.35])
	graph_axes.append([0.00, 0.6, 0.28, 0.35])
	graph_axes.append([0.34, 0.6, 0.28, 0.35])
	graph_axes.append([0.67, 0.6, 0.28, 0.35])

	slider_axes.append([0.07, 0.01, 0.15, 0.03])
	slider_axes.append([0.41, 0.01, 0.15, 0.03])
	slider_axes.append([0.75, 0.01, 0.15, 0.03])
	slider_axes.append([0.07, 0.51, 0.15, 0.03])
	slider_axes.append([0.41, 0.51, 0.15, 0.03])
	slider_axes.append([0.75, 0.51, 0.15, 0.03])

	button_axes.append([[0.01, 0.01, 0.02, 0.03], [0.04, 0.01, 0.02, 0.03]])
	button_axes.append([[0.35, 0.01, 0.02, 0.03], [0.38, 0.01, 0.02, 0.03]])
	button_axes.append([[0.69, 0.01, 0.02, 0.03], [0.72, 0.01, 0.02, 0.03]])
	button_axes.append([[0.01, 0.51, 0.02, 0.03], [0.04, 0.51, 0.02, 0.03]])
	button_axes.append([[0.35, 0.51, 0.02, 0.03], [0.38, 0.51, 0.02, 0.03]])
	button_axes.append([[0.69, 0.51, 0.02, 0.03], [0.72, 0.51, 0.02, 0.03]])

	save_axes.append([0.25, 0.01, 0.02, 0.03])
	save_axes.append([0.59, 0.01, 0.02, 0.03])
	save_axes.append([0.93, 0.01, 0.02, 0.03])
	save_axes.append([0.25, 0.51, 0.02, 0.03])
	save_axes.append([0.59, 0.51, 0.02, 0.03])
	save_axes.append([0.93, 0.51, 0.02, 0.03])

	save_all_axes = [0.93, 0.61, 0.04, 0.06]

	# Creating graph display
	display_graph.append(fig.add_axes(graph_axes[0]))
	display_graph.append(fig.add_axes(graph_axes[1]))
	display_graph.append(fig.add_axes(graph_axes[2]))
	display_graph.append(fig.add_axes(graph_axes[3]))
	display_graph.append(fig.add_axes(graph_axes[4]))
	display_graph.append(fig.add_axes(graph_axes[5]))

	display_slider.append(fig.add_axes(slider_axes[0]))
	display_slider.append(fig.add_axes(slider_axes[1]))
	display_slider.append(fig.add_axes(slider_axes[2]))
	display_slider.append(fig.add_axes(slider_axes[3]))
	display_slider.append(fig.add_axes(slider_axes[4]))
	display_slider.append(fig.add_axes(slider_axes[5]))

	display_button.append([fig.add_axes(button_axes[0][0]), fig.add_axes(button_axes[0][1])])
	display_button.append([fig.add_axes(button_axes[1][0]), fig.add_axes(button_axes[1][1])])
	display_button.append([fig.add_axes(button_axes[2][0]), fig.add_axes(button_axes[2][1])])
	display_button.append([fig.add_axes(button_axes[3][0]), fig.add_axes(button_axes[3][1])])
	display_button.append([fig.add_axes(button_axes[4][0]), fig.add_axes(button_axes[4][1])])
	display_button.append([fig.add_axes(button_axes[5][0]), fig.add_axes(button_axes[5][1])])

	display_save.append(fig.add_axes(save_axes[0]))
	display_save.append(fig.add_axes(save_axes[1]))
	display_save.append(fig.add_axes(save_axes[2]))
	display_save.append(fig.add_axes(save_axes[3]))
	display_save.append(fig.add_axes(save_axes[4]))
	display_save.append(fig.add_axes(save_axes[5]))

	save_all_display = fig.add_axes(save_all_axes)

	# Showing graph display
	display_graph[0].imshow(img_abs[0, ::, ::], cmap = 'gray')
	display_graph[1].imshow(img_abs[::, 0, ::], cmap = 'gray')
	display_graph[2].imshow(img_abs[::, ::, 0], cmap = 'gray')
	display_graph[3].imshow(img_scat[0, ::, ::], cmap = 'gray')
	display_graph[4].imshow(img_scat[::, 0, ::], cmap = 'gray')
	display_graph[5].imshow(img_scat[::, ::, 0], cmap = 'gray')

	slider.append(Slider(display_slider[0], "", valmin = 0, valmax = 275, valinit = initializer, valstep = 1.0))
	slider.append(Slider(display_slider[1], "", valmin = 0, valmax = 275, valinit = initializer, valstep = 1.0))
	slider.append(Slider(display_slider[2], "", valmin = 0, valmax = 275, valinit = initializer, valstep = 1.0))
	slider.append(Slider(display_slider[3], "", valmin = 0, valmax = 275, valinit = initializer, valstep = 1.0))
	slider.append(Slider(display_slider[4], "", valmin = 0, valmax = 275, valinit = initializer, valstep = 1.0))
	slider.append(Slider(display_slider[5], "", valmin = 0, valmax = 275, valinit = initializer, valstep = 1.0))

	button.append([Button(display_button[0][0], '$\u25C0$'), Button(display_button[0][1], '$\u25B6$')])
	button.append([Button(display_button[1][0], '$\u25C0$'), Button(display_button[1][1], '$\u25B6$')])
	button.append([Button(display_button[2][0], '$\u25C0$'), Button(display_button[2][1], '$\u25B6$')])
	button.append([Button(display_button[3][0], '$\u25C0$'), Button(display_button[3][1], '$\u25B6$')])
	button.append([Button(display_button[4][0], '$\u25C0$'), Button(display_button[4][1], '$\u25B6$')])
	button.append([Button(display_button[5][0], '$\u25C0$'), Button(display_button[5][1], '$\u25B6$')])

	save.append(Button(display_save[0], label = '\u25CF'))
	save.append(Button(display_save[1], label = '\u25CF'))
	save.append(Button(display_save[2], label = '\u25CF'))
	save.append(Button(display_save[3], label = '\u25CF'))
	save.append(Button(display_save[4], label = '\u25CF'))
	save.append(Button(display_save[5], label = '\u25CF'))

	save_all_button = Button(save_all_display, label = 'save\nall')

	# Updating
	# Link sliders
	def update_0(val):
		display_graph[0].imshow(img_abs[int(slider[0].val), ::, ::], cmap = 'gray')
		fig.canvas.draw_idle()

	def update_1(val):
		display_graph[1].imshow(img_abs[::, int(slider[1].val), ::], cmap = 'gray')
		fig.canvas.draw_idle()

	def update_2(val):
		display_graph[2].imshow(img_abs[::, ::, int(slider[2].val)], cmap = 'gray')
		fig.canvas.draw_idle()

	def update_3(val):
		display_graph[3].imshow(img_scat[int(slider[3].val), ::, ::], cmap = 'gray')
		fig.canvas.draw_idle()

	def update_4(val):
		display_graph[4].imshow(img_scat[::, int(slider[4].val), ::], cmap = 'gray')
		fig.canvas.draw_idle()

	def update_5(val):
		display_graph[5].imshow(img_scat[::, ::, int(slider[5].val)], cmap = 'gray')
		fig.canvas.draw_idle()

	display_graph[0].set_title("Absorption - YX")
	display_graph[1].set_title("Absorption - ZX")
	display_graph[2].set_title("Absorption - ZY")
	display_graph[3].set_title("Scattering - YX")
	display_graph[4].set_title("Scattering - ZX")
	display_graph[5].set_title("Scattering - ZY")

	# Linking buttons
	def forward_0(val):
		slider[0].set_val(slider[0].val + 1)

	def forward_1(val):
		slider[1].set_val(slider[1].val + 1)

	def forward_2(val):
		slider[2].set_val(slider[2].val + 1)

	def forward_3(val):
		slider[3].set_val(slider[3].val + 1)

	def forward_4(val):
		slider[4].set_val(slider[4].val + 1)

	def forward_5(val):
		slider[5].set_val(slider[5].val + 1)

	def backward_0(val):
		slider[0].set_val(slider[0].val - 1)

	def backward_1(val):
		slider[1].set_val(slider[1].val - 1)

	def backward_2(val):
		slider[2].set_val(slider[2].val - 1)

	def backward_3(val):
		slider[3].set_val(slider[3].val - 1)

	def backward_4(val):
		slider[4].set_val(slider[4].val - 1)

	def backward_5(val):
		slider[5].set_val(slider[5].val - 1)

	# Saving imgs

	# Directorios
	def check_create_dir():
		save_dir = Path('./save')
		if save_dir.exists() != True:
			save_dir.mkdir()
			print("Directory './save' was created successfully. All images will be saved inside")
		else:
			print("Directory './save' already exists. All images will be saved inside")

	# def printing_arrow(event):
	# 	display_graph[0].arrow(event.xdata - 20, event.ydata + 30, 20, -30, width = 2, color = 'white')
	# 	print('You pressed at: ', event.xdata, event.ydata)

	# cid = fig.canvas.mpl_connect('button_press_event', printing_arrow)

	def save_axes_figure_0(val):
		name = str(int(slider[0].val))
		zl = np.array(display_graph[0].get_xbound(), dtype = np.int)
		yl = np.array(display_graph[0].get_ybound(), dtype = np.int)
		plt.imsave(fname = f"save/abs.yx.{name}.png", arr = img_abs[int(slider[0].val), yl[0] : yl[1], zl[0] : zl[1]], cmap = 'gray')

	def save_axes_figure_1(val):
		name = str(int(slider[1].val))
		zl = np.array(display_graph[1].get_xbound(), dtype = np.int)
		xl = np.array(display_graph[1].get_ybound(), dtype = np.int)
		plt.imsave(fname = f"save/abs.zx.{name}.png", arr = img_abs[xl[0] : xl[1], int(slider[1].val), zl[0] : zl[1]], cmap = 'gray')

	def save_axes_figure_2(val):
		name = str(int(slider[2].val))
		yl = np.array(display_graph[2].get_xbound(), dtype = np.int)
		xl = np.array(display_graph[2].get_ybound(), dtype = np.int)
		plt.imsave(fname = f"save/abs.zy.{name}.png", arr = img_abs[xl[0] : xl[1], yl[0] : yl[1], int(slider[2].val)], cmap = 'gray')

	def save_axes_figure_3(val):
		name = str(int(slider[3].val))
		zl = np.array(display_graph[3].get_xbound(), dtype = np.int)
		yl = np.array(display_graph[3].get_ybound(), dtype = np.int)
		plt.imsave(fname = f"save/scat.yx.{name}.png", arr = img_scat[int(slider[3].val), yl[0] : yl[1], zl[0] : zl[1]], cmap = 'gray')

	def save_axes_figure_4(val):
		name = str(int(slider[4].val))
		zl = np.array(display_graph[4].get_xbound(), dtype = np.int)
		xl = np.array(display_graph[4].get_ybound(), dtype = np.int)
		plt.imsave(fname = f"save/scat.zx.{name}.png", arr = img_scat[xl[0] : xl[1], int(slider[4].val), zl[0] : zl[1]], cmap = 'gray')

	def save_axes_figure_5(val):
		name = str(int(slider[5].val))
		yl = np.array(display_graph[5].get_xbound(), dtype = np.int)
		xl = np.array(display_graph[5].get_ybound(), dtype = np.int)
		plt.imsave(fname = f"save/scat.zy.{name}.png", arr = img_scat[xl[0] : xl[1], yl[0] : yl[1], int(slider[3].val)], cmap = 'gray')

	# Saving all imgs
	def save_all(val):
		font = {'family': 'Sans-Serif', 'weight' : 'bold', 'size' : 12}
		font_title = {'family': 'Sans-Serif', 'weight' : 'bold', 'size' : 15}
		plt.rc('font', **font)
		fig_save, ax = plt.subplots(ncols = 3, nrows = 2, figsize = (8, 6), dpi = 100, constrained_layout = True)

		zl0 = np.array(display_graph[0].get_xbound(), dtype = np.int)
		yl0 = np.array(display_graph[0].get_ybound(), dtype = np.int)

		zl1 = np.array(display_graph[1].get_xbound(), dtype = np.int)
		xl1 = np.array(display_graph[1].get_ybound(), dtype = np.int)

		yl2 = np.array(display_graph[2].get_xbound(), dtype = np.int)
		xl2 = np.array(display_graph[2].get_ybound(), dtype = np.int)

		zl3 = np.array(display_graph[3].get_xbound(), dtype = np.int)
		yl3 = np.array(display_graph[3].get_ybound(), dtype = np.int)

		zl4 = np.array(display_graph[4].get_xbound(), dtype = np.int)
		xl4 = np.array(display_graph[4].get_ybound(), dtype = np.int)

		yl5 = np.array(display_graph[5].get_xbound(), dtype = np.int)
		xl5 = np.array(display_graph[5].get_ybound(), dtype = np.int)

		ax[0][0].imshow(img_scat[int(slider[3].val), yl3[0] : yl3[1], zl3[0] : zl3[1]], cmap = 'gray')
		ax[0][1].imshow(img_scat[xl4[0] : xl4[1], int(slider[4].val), zl4[0] : zl4[1]], cmap = 'gray')
		ax[0][2].imshow(img_scat[xl5[0] : xl5[1], yl5[0] : yl5[1], int(slider[5].val)], cmap = 'gray')
		ax[1][0].imshow(img_abs[int(slider[0].val), yl0[0] : yl0[1], zl0[0] : zl0[1]], cmap = 'gray')
		ax[1][1].imshow(img_abs[xl1[0] : xl1[1], int(slider[1].val), zl1[0] : zl1[1]], cmap = 'gray')
		ax[1][2].imshow(img_abs[xl2[0] : xl2[1], yl2[0] : yl2[1], int(slider[2].val)], cmap = 'gray')

		ax[0][0].text(7, 20, "YX", color = 'white', backgroundcolor = '0.2')
		ax[0][0].set_title("Absorption", font = font_title)
		ax[0][1].text(7, 20, "ZX", color = 'white', backgroundcolor = '0.2')
		ax[0][1].set_title("Absorption", font = font_title)
		ax[0][2].text(7, 20, "ZY", color = 'white', backgroundcolor = '0.2')
		ax[0][2].set_title("Absorption", font = font_title)
		ax[1][0].text(7, 20, "YX", color = 'white', backgroundcolor = '0.2')
		ax[1][0].set_title("Scattering", font = font_title)
		ax[1][1].text(7, 20, "ZX", color = 'white', backgroundcolor = '0.2')
		ax[1][1].set_title("Scattering", font = font_title)
		ax[1][2].text(7, 20, "ZY", color = 'white', backgroundcolor = '0.2')
		ax[1][2].set_title("Scattering", font = font_title)

		ax[0][0].axis("off")
		ax[0][1].axis("off")
		ax[0][2].axis("off")
		ax[1][0].axis("off")
		ax[1][1].axis("off")
		ax[1][2].axis("off")

		path = Path('save/all.png')
		if path.exists and path.is_file():
			print("The file already exists, please write a name for new file, or type 0 to replace the existent file")
			name = input()
			fig_save.savefig(f"save/{name}.png")
			print(f"{name}.png was created successfully")
		else:
			fig_save.savefig("save/all.png")
			print("all.png was created successfully")

	# Done
	# update(0)
	slider[0].on_changed(update_0)
	slider[1].on_changed(update_1)
	slider[2].on_changed(update_2)
	slider[3].on_changed(update_3)
	slider[4].on_changed(update_4)
	slider[5].on_changed(update_5)

	button[0][0].on_clicked(backward_0)
	button[0][1].on_clicked(forward_0)
	button[1][0].on_clicked(backward_1)
	button[1][1].on_clicked(forward_1)
	button[2][0].on_clicked(backward_2)
	button[2][1].on_clicked(forward_2)
	button[3][0].on_clicked(backward_3)
	button[3][1].on_clicked(forward_3)
	button[4][0].on_clicked(backward_4)
	button[4][1].on_clicked(forward_4)
	button[5][0].on_clicked(backward_5)
	button[5][1].on_clicked(forward_5)

	save[0].on_clicked(save_axes_figure_0)
	save[1].on_clicked(save_axes_figure_1)
	save[2].on_clicked(save_axes_figure_2)
	save[3].on_clicked(save_axes_figure_3)
	save[4].on_clicked(save_axes_figure_4)
	save[5].on_clicked(save_axes_figure_5)

	save_all_button.on_clicked(save_all)

	check_create_dir()
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'This script allows viewing, parsing and saving images from 3D tiff and tif files')
	parser.add_argument('filename_abs', type = str, help = 'name of absorption file to open')
	parser.add_argument('filename_scat', type = str, help = 'name of scattering file to open')
	args = parser.parse_args()
	slicing_3D_tif_tiff_file(args.filename_abs, args.filename_scat)

