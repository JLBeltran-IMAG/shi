import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tifffile as ti
import tomopy

from skimage import exposure
from skimage import io

from pathlib import Path

import argparse

# Programar script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT Image Viewer")
    parser.add_argument("--path", type=str, help="Path to the folder containing the images")
    parser.add_argument("--contrast", type=str, help="Contrast type of the images")
    parser.add_argument("--orientation", type=str, help="Orientation of the images")

    args = parser.parse_args()

    main_path = args.path
    contrast = args.contrast
    orientation = args.orientation

    path_to_ctscan = f"{main_path}/{contrast}/flat_corrections/{orientation}"
    path_to_angles = f"{main_path}"

    images_path = sorted([filepath for filepath in Path(path_to_ctscan).glob("*.tif")])
    images = np.array([ti.imread(imgpath) for imgpath in images_path])

    theta_list = list()
    ndeg = 1e-9

    with open(f"{path_to_angles}/positions.txt", "r") as file_positions:
        for i in file_positions.readlines():
            angle = ndeg * int(i.strip())
            theta_list.append(angle)

    theta = np.deg2rad(np.array(theta_list, dtype=np.float32))
    # ----------------------------------------


    # ==================================================================================
    # Número de bins para el histograma
    bins = 300

    # Crear la figura con dos columnas (imagen y histograma)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(16, 6))

    # Divisores para agregar sliders debajo de los ejes
    divider_imgs = make_axes_locatable(ax_img)
    divider_hist = make_axes_locatable(ax_hist)

    # Agregar espacio para sliders debajo de los ejes
    ax_slider_imgs = divider_imgs.append_axes("bottom", size="5%", pad=0.2)
    ax_button_save = divider_imgs.append_axes("bottom", size="5%", pad=0.1)
    ax_button_cbct = divider_imgs.append_axes("bottom", size="5%", pad=0.1)
    ax_slider_xmin = divider_hist.append_axes("bottom", size="5%", pad=0.6)
    ax_slider_xmax = divider_hist.append_axes("bottom", size="5%", pad=0.0)

    # Mostrar la primera imagen
    img_display = ax_img.imshow(images[0], cmap='gray')
    ax_img.set_title(f"Angle of projection {theta[0]:.3f}")
    # ax_img.axis('off')

    # Inicializar el histograma de la primera imagen
    hist, bin_edges = np.histogram(images[0], bins=bins, density=True)
    line, = ax_hist.plot(bin_edges[:-1], hist, color='blue', lw=2)  # Línea inicial

    ax_hist.set_xlim(bin_edges.min(), bin_edges.max())
    ax_hist.set_ylim(0, hist.max() * 1.1)
    ax_hist.set_title("Histograma")
    ax_hist.set_xlabel("Intensidad de píxel")
    ax_hist.set_ylabel("Frecuencia normalizada")

    # Horizontal line for centering
    vline = ax_img.axvline(x=2, color='red', linestyle='--', linewidth=2)  # Línea inicial

    # Sliders
    val_min = bin_edges.min()
    val_max = bin_edges.max()

    # Slider para la imagen
    slider_img = Slider(ax_slider_imgs, 'Images', 0, len(images) - 1, valinit=0, valstep=1)

    # Sliders para `xmin` y `xmax`
    slider_xmin = Slider(ax_slider_xmin, 'x-min', val_min, val_max, valinit=val_min)
    slider_xmax = Slider(ax_slider_xmax, 'x-max', val_min, val_max, valinit=val_max)


    # Button to export the images processed
    button_save_manual_adj = Button(ax_button_save, 'Export manual adjustment')
    button_save_auto_adj = Button(ax_button_cbct, 'Export automatic adjustment')


    # Función para actualizar la imagen y el histograma
    def update_img(val):
        idx = int(slider_img.val)  # Obtén el índice del slider_img
        img_display.set_array(images[idx])  # Actualiza la imagen
        ax_img.set_title(f"Angle of projection {theta[idx]:.3f}")

        img_display.set_clim(slider_xmin.val, slider_xmax.val)

        # Calcular y actualizar el histograma
        hist, bin_edges = np.histogram(images[idx], bins=bins, density=True)
        line.set_data(bin_edges[:-1], hist)

        # Ajustar el eje y del histograma
        ax_hist.set_ylim(0, hist.max() * 1.1)

        fig.canvas.draw_idle()  # Redibujar la figura


    # Función para actualizar los límites del eje x del histograma
    def update_xlim(val):
        xmin = slider_xmin.val  # Límite inferior
        xmax = slider_xmax.val  # Límite superior

        img_display.set_clim(xmin, xmax)

        ax_hist.set_xlim(xmin, xmax)
        fig.canvas.draw_idle()  # Redibujar la figura


    # Variable de estado para controlar si el movimiento está activado
    is_active = False

    # Función para mover la línea vertical con el mouse
    def on_click(event):
        global is_active
        if event.inaxes == ax_img:  # Verificar si el clic fue en el eje de la imagen
            is_active = not is_active  # Cambiar el estado (activar/desactivar)

            vline.set_xdata([event.xdata])  # Mover la línea vertical a la posición del mouse
            fig.canvas.draw_idle()  # Redibujar la figura


    def on_mouse_move(event):
        if is_active and event.inaxes == ax_img:  # Verificar si está activado y el mouse está en el eje
            if event.xdata is not None:  # Asegurarse de que hay datos de posición válidos
                vline.set_xdata([event.xdata])  # Mover la línea vertical a la posición del mouse
                fig.canvas.draw_idle()  # Redibujar la figura


    # Función para manejar la desactivación del movimiento al soltar el botón del mouse
    def on_release(event):
        global is_active
        if is_active:  # Desactivar el movimiento cuando se suelta el clic
            is_active = False


    # Función para exportar la imagen actual
    def export_image_manual(event):
        dir_to_export = Path(f"{main_path}/ct/{contrast}/{orientation}/manual_adjusted")
        dir_to_export.mkdir(parents=True, exist_ok=True)

        for img, path_to_export in zip(images, images_path):
            saving2 = dir_to_export.joinpath(path_to_export.name)
            img_to_export = exposure.rescale_intensity(img, in_range=(slider_xmin.val, slider_xmax.val))
            io.imsave(saving2, img_to_export)
        
        export_centers(exposure.rescale_intensity(images, in_range=(slider_xmin.val, slider_xmax.val)))

        print("Export DONE")


    def export_image_auto(event):
        dir_to_export = Path(f"{main_path}/ct/{contrast}/{orientation}/auto_adjusted")
        dir_to_export.mkdir(parents=True, exist_ok=True)

        for img, path_to_export in zip(images, images_path):
            p2, p98 = np.percentile(img, (2, 98))
            saving2 = dir_to_export.joinpath(path_to_export.name)
            img_to_export = exposure.rescale_intensity(img, in_range=(p2, p98))
            io.imsave(saving2, img_to_export)
        
        export_centers(exposure.rescale_intensity(images, in_range=(slider_xmin.val, slider_xmax.val)))

        print("Export DONE")


    def export_centers(images):
        # -------------------------- Defining rotation centers manually -------------------------
        center = vline.get_xdata()[0]
        start = int(center - 10)
        stop = int(center + 10)
        manual_center = np.linspace(start, stop, 20, endpoint=True)

        # -------------------------- Defining rotation centers automately -----------------------
        center = tomopy.find_center(images, theta)[0]
        start = int(center - 10)
        stop = int(center + 10)
        center_nm_entropy = np.linspace(start, stop, 20, endpoint=True)

        center = tomopy.find_center_vo(images)
        start = int(center - 10)
        stop = int(center + 10)
        center_nghia_vos_method = np.linspace(start, stop, 20, endpoint=True)

        # -------------------------- Exporting centers to file ----------------------------------
        np.savez(f"{main_path}/ct/{contrast}/{orientation}/centers", manual_center, center_nm_entropy, center_nghia_vos_method)


    # Conectar el botón a la función de exportación
    button_save_manual_adj.on_clicked(export_image_manual)
    button_save_auto_adj.on_clicked(export_image_auto)

    # Conectar los sliders a sus funciones
    slider_img.on_changed(update_img)
    slider_xmin.on_changed(update_xlim)
    slider_xmax.on_changed(update_xlim)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # Ajustar el diseño
    plt.tight_layout()
    plt.show()


