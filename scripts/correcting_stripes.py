import numpy as np
import tifffile
from PySide6.QtWidgets import QApplication, QFileDialog
from pathlib import Path


def delete_detector_stripes(image, stripe_rows, stripe_cols):
    """
    Remueve las filas y columnas (stripes) especificadas de la imagen.

    Parameters:
        image (np.ndarray): La imagen de entrada como arreglo de NumPy.
        stripe_rows (list): Lista de índices de filas a eliminar.
        stripe_cols (list): Lista de índices de columnas a eliminar.

    Returns:
        np.ndarray: La imagen con los stripes eliminados.
    """
    # Eliminar las filas especificadas
    image_clean = np.delete(image, stripe_rows, axis=0)

    # Eliminar las columnas especificadas
    image_clean = np.delete(image_clean, stripe_cols, axis=1)
    return image_clean


def correcting_stripes():
    """
    Solicita al usuario seleccionar una carpeta y procesa todos los archivos TIFF encontrados
    (excluyendo aquellos en directorios cuyo nombre contenga 'no_stripe'). Para cada imagen,
    se eliminan los stripes especificados y se sobrescribe el archivo original con la imagen corregida.
    """
    app = QApplication([])
    default_folder = Path.home() / "Documents" / "CXI" / "CXI-DATA-ACQUISITION"
    folder = QFileDialog.getExistingDirectory(None, "Elige una carpeta", str(default_folder))

    # Verificar si el usuario canceló la selección
    if not folder:
        print("No se seleccionó ninguna carpeta. Saliendo...")
        return

    app.quit()

    base_path = Path(folder)
    # Recopilar los directorios que contienen archivos TIFF
    image_dirs = {path.parent for path in base_path.rglob("*.tif")}

    # Definir los índices de filas y columnas a eliminar
    stripe_rows = [2944, 2945]
    stripe_cols = [295, 722, 1167, 1388, 1541, 2062, 2302, 2303]

    for directory in image_dirs:
        for file_path in directory.glob("*.tif"):
            try:
                # Leer la imagen TIFF
                image = tifffile.imread(file_path)

                # Eliminar los stripes especificados
                image_clean = delete_detector_stripes(image, stripe_rows, stripe_cols)
                
                # Sobrescribir el archivo original con la imagen corregida
                tifffile.imwrite(file_path, image_clean, imagej=True)
                print(f"Procesado: {file_path}")

            except Exception as e:
                print(f"Error procesando {file_path}: {e}")


if __name__ == "__main__":
    correcting_stripes()
