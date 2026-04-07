#!/usr/bin/env python3

import numpy as np
from skimage import io
from skimage.exposure import rescale_intensity as norm_img
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import ctypes
import argparse
import sys
import logging
import time
from pathlib import Path
import logging as log
import smaract.ctl as ctl


# -------#
# Source #
# -------#
src_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_dir))
# --------

# from detector_control import configure_safe_base, configure_ext_trigger
import acq_directories
import detector_control
import motor_control
import msgs
import pygigev

msgs_ = msgs.messages()


# =======================================================================
# =============================   Parser   ==============================
def my_parser():
    parser = argparse.ArgumentParser(
        prog="ACQ",
        description="%(prog)s: This software is an automated implementation for taking_nimgs images with order",
    )

    # Defining arguments for various functionalities
    parser.add_argument("-n", "--name", required=True, type=str, help="Name of the directory.")
    # parser.add_argument("-d", "--delete", type=str, help="Delete the directory with the name specify by -d or --delete")

    return parser


# =======================================================================
# ============================   VIEWER CONTROL   =======================
class AcqViewerApp:
    """A class representing the main GUI application for SHI-DAQ Viewer, designed for image acquisition and viewing.

    This class provides a graphical interface for controlling image acquisition from a detector,
    displaying live previews, and saving images in different modes (flat, dark, bright, sample,
    and computed tomography).

    Attributes:
        handle: Device handle for detector communication
        payload_size: Size of image payload in bytes
        pixel_format: Format of image pixels
        pixel_format_unpacked: Unpacked format of image pixels
        width_str: Width of image in pixels
        height_str: Height of image in pixels
        root: Main tkinter window
        main_frame: Main container frame
        image_frame: Frame containing image display and controls
        image_label: Label widget displaying the current image
        _update_active: Boolean flag for continuous image updating
        _project_dir: Path to project directory
        _dark_dir: Path to dark images directory
        _flat_dir: Path to flat images directory
        _bright_dir: Path to bright images directory
        _sample_dir: Path to sample images directory

    Methods:
        _snap_image(): Captures a single image from the detector
        _save_snap(): Saves the current image to disk
        _toggle_grab(): Toggles continuous image acquisition
        _init_image(): Initializes the image display
        _action_flat(): Acquires flat field images
        _action_dark(): Acquires dark field images
        _action_bright(): Acquires bright field images
        _action_sample(): Acquires sample images
        _start_ct_acquisition(): Starts computed tomography acquisition
        on_closing(): Handles application closure

    The class manages both the GUI elements and the underlying image acquisition functionality,
    providing a complete interface for X-ray imaging operations. It supports various acquisition
    modes including single shots, multiple images, and computed tomography sequences."""

    def __init__(self, root, directory):
        """
        Initializes the main graphical user interface (GUI) for the SHI-DAQ Viewer application.

        Parameters:
        -----------
        root : tkinter.Tk
            The root window for the application GUI.
        directory : tuple
            A tuple containing the directory paths for project files, including:
            - project_dir : str : Path for the main project directory
            - dark_dir : str : Path for storing dark images
            - flat_dir : str : Path for storing flat images
            - bright_dir : str : Path for storing bright images
            - sample_dir : str : Path for storing sample images
        handle : object
            The device or image acquisition handle needed for capturing images.
        numBuffers : int
            The number of image buffers to use in acquisition.


        Notes:
        ------
        This initialization method configures the main components of the GUI layout, sets up control buttons for image acquisition,
        and initializes an image preview with the `_snap_image` method.
        """
        # init detector
        (self.handle, self.payload_size, self.pixel_format, self.pixel_format_unpacked, self.width_str, self.height_str) = (
            detector_control.starting_detector()
        )

        detector_control.configure_safe_base(self.handle)


        # directories
        (self._project_dir, self._dark_dir, self._flat_dir, self._bright_dir, self._sample_dir) = directory

        # graphical user interface
        self.root = root
        self.root.title("SHI-DAQ Viewer")

        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        # Image frame
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.grid(row=0, column=0, sticky="nsew")

        # Image viewer and control
        self._update_active = False

        snap_frame = tk.Frame(self.image_frame)
        snap_frame.pack(pady=10, padx=10, fill="x")
        #
        self.snap_button = tk.Button(snap_frame, text="Snap", command=self._snap_image)
        self.snap_button.pack(padx=(0, 10), side="left")
        #
        self.save_button = tk.Button(snap_frame, text="Save", command=self._save_snap)
        self.save_button.pack(padx=(0, 10), side="left")
        #
        self.exp_time = tk.Label(snap_frame, text="Exposure time")
        self.exp_time.pack(padx=(10, 5), side="left")
        #
        self.exp_time_value = tk.Spinbox(snap_frame, from_=1000, to=15000, width=6)
        self.exp_time_value.pack(side="left")

        # ----------
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(padx=10, pady=(10, 10), fill="both", expand=True)
        #
        # init first image to show before acquisition
        self.image = np.zeros((int(self.height_str.value), int(self.width_str.value)), dtype=np.uint16)
        self.image = np.zeros((int(700.0), int(500.0)), dtype=np.uint16)
        self._init_image()

        # Buttons
        self.acquisition_frame = tk.Frame(self.main_frame)
        self.acquisition_frame.grid(row=0, column=1, padx=10, sticky="n")
        self._create_acquisition_projection()
        self._create_acquisition_tomography()

        # Init images
        # self._snap_image()

        # Close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def on_closing(self):
        if messagebox.askyesno("Confirmation", "Are you sure you want to close the application?"):
            # detector_control.stopping_transfer_and_detector(self.handle)
            self.root.destroy()

    # ---------------------- Image viewer control ----------------------
    def _toggle_grab(self):
        # pass
        # Turn update
        self._update_active = not self._update_active

        if self._update_active:
            self.grab_button.config(text="Stop")
            self._grab_image()
        else:
            self.grab_button.config(text="Grab")

    def _init_image(self):
        # pass
        # Obtener la imagen
        self.pil_image = Image.fromarray(self.image)

        width, height = self.pil_image.size
        width = int(width * 0.15)
        height = int(height * 0.15)

        self.pil_image = self.pil_image.resize((width, height))

        # Convertir a formato Tkinter
        tk_image = ImageTk.PhotoImage(self.pil_image)

        # Actualizar la etiqueta con la imagen
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

    def _snap_image(self):
        # pass
        numBuffers = 2
        # Allocate buffers to store images in (2 here).
        buffer_addresses = ((ctypes.c_void_p) * numBuffers)()

        # (Handle cases where image is larger than payload due to pixel unpacking)
        bufsize = self.payload_size.value
        bufsize_unpacked = (
            int(self.width_str.value)
            * int(self.height_str.value)
            * pygigev.GevGetPixelSizeInBytes(self.pixel_format_unpacked)
        )
        if bufsize_unpacked > bufsize:
            bufsize = bufsize_unpacked

        for bufIndex in range(numBuffers):
            temp = ((ctypes.c_char) * bufsize)()
            buffer_addresses[bufIndex] = ctypes.cast(temp, ctypes.c_void_p)

        # Initialize a transfer (Asynchronous cycling)
        status = pygigev.GevInitializeTransfer(
            self.handle, pygigev.Asynchronous, self.payload_size, numBuffers, buffer_addresses
        )

        # Grab images to fill the buffers
        numImages = numBuffers
        status = pygigev.GevStartTransfer(self.handle, numImages)

        # Read the images out
        gevbufPtr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()

        for imgIndex in range(numImages):
            tmout = (ctypes.c_uint32)(30000)

            status = pygigev.GevWaitForNextFrame(self.handle, ctypes.byref(gevbufPtr), tmout.value)

            if status != 0:
                continue

            # Check img data status
            gevbuf = gevbufPtr.contents
            if gevbuf.status != 0:
                print("Img #", imgIndex, "ERROR status =", gevbuf.status)
                continue

            print("Img #", imgIndex, "SUCCESS status =", gevbuf.status)
            # Make a PIL image out of this frame
            im_addr = ctypes.cast(gevbuf.address, ctypes.POINTER(ctypes.c_ubyte * gevbuf.recv_size))

            image = np.frombuffer(im_addr.contents, dtype=np.uint16).reshape(
                (int(self.height_str.value), int(self.width_str.value))
            )

            # showing image
            pil_image = Image.fromarray(norm_img(image, out_range=(0, 255)))
            width, height = pil_image.size
            width = int(width * 0.15)
            height = int(height * 0.15)
            pil_image = pil_image.resize((width, height))
            tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image
            self.image_label.update_idletasks()

        # Free the transfer
        status = pygigev.GevStopTransfer(self.handle)
        if status == 0:
            print("Free transfer done")

        else:
            print("Free transfer error")

        status = pygigev.GevFreeTransfer(self.handle)
        if status == 0:
            print("Free transfer done")

        else:
            print("Free transfer error")

    def _save_snap(self):
        # pass
        save_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif")])

        if save_path:
            self.pil_image.save(f"{save_path}", compression="none")
            messagebox.showinfo("Success", "Flat images were successfully saved")

        else:
            messagebox.showwarning("Warning", "Image was not saved")

    # ---------------------- Defining buttons ----------------------
    def _create_acquisition_projection(self):
        # Function 2 => DARK
        # ------------------
        row_frame2 = tk.Frame(self.acquisition_frame)
        row_frame2.pack(pady=5, fill="x")
        # -------------
        button2 = tk.Button(row_frame2, text="Dark image(s)", width=20, command=self._action_dark)
        button2.pack(side="left")
        # -------------
        self.spinbox2 = tk.Spinbox(row_frame2, from_=1, to=10000, width=5)
        self.spinbox2.pack(side="left", padx=5)
        # -------------
        self.text_box2 = tk.Text(row_frame2, width=15, height=1, fg="grey", state="normal")
        self.text_box2.pack(side="left", padx=5)
        self.text_box2.insert("1.0", "dark")
        self.text_box2.configure(state="disabled")
        # --------------------------------------------------------------

        # Function 3 => BRIGHT
        # --------------------
        row_frame3 = tk.Frame(self.acquisition_frame)
        row_frame3.pack(pady=5, fill="x")
        # -------------
        button3 = tk.Button(row_frame3, text="Bright image(s)", width=20, command=self._action_bright)
        button3.pack(side="left")
        # -------------
        self.spinbox3 = tk.Spinbox(row_frame3, from_=1, to=10000, width=5)
        self.spinbox3.pack(side="left", padx=5)
        # -------------
        self.text_box3 = tk.Text(row_frame3, width=15, height=1, fg="grey", state="normal")
        self.text_box3.pack(side="left", padx=5)
        self.text_box3.insert("1.0", "bright")
        self.text_box3.configure(state="disabled")
        # --------------------------------------------------------------

        # Function 1 => FLAT
        # ------------------
        row_frame1 = tk.Frame(self.acquisition_frame)
        row_frame1.pack(pady=5, fill="x")
        # -------------
        button1 = tk.Button(row_frame1, text="Flat image(s)", width=20, command=self._action_flat)
        button1.pack(side="left")
        # -------------
        self.spinbox1 = tk.Spinbox(row_frame1, from_=1, to=10000, width=5)
        self.spinbox1.pack(side="left", padx=5)
        # -------------
        self.text_box1 = tk.Text(row_frame1, width=15, height=1, fg="grey", state="normal")
        self.text_box1.pack(side="left", padx=5)
        self.text_box1.insert("1.0", "flat")
        self.text_box1.configure(state="disabled")
        # --------------------------------------------------------------

        # Function 4 => SAMPLE
        # --------------------
        row_frame4 = tk.Frame(self.acquisition_frame)
        row_frame4.pack(pady=5, fill="x")
        # -------------
        button4 = tk.Button(row_frame4, text="Sample image(s)", width=20, command=self._action_sample)
        button4.pack(side="left")
        # -------------
        self.spinbox4 = tk.Spinbox(row_frame4, from_=1, to=10000, width=5)
        self.spinbox4.pack(side="left", padx=5)
        # -------------
        self.text_box4 = tk.Text(row_frame4, width=15, height=1)
        self.text_box4.pack(side="left", padx=5)
        self.text_box4.insert("1.0", "sample-name")
        # --------------------------------------------------------------

    def _create_acquisition_tomography(self):
        # Function 1 => COMPUTED TOMOGRAPHY - IMAGES
        # ------------------
        ct_frame = tk.Frame(self.acquisition_frame)
        ct_frame.pack(pady=(50, 5), fill="x")
        # -------------
        ct_label = tk.Label(ct_frame, text="Computed tomography", width=20)
        ct_label.pack(side="left")
        # -------------
        self.spinbox_ct = tk.Spinbox(ct_frame, from_=100, to=99999, width=8, validate="key")
        self.spinbox_ct.pack(side="left", padx=5)
        # -------------
        self.text_box_ct = tk.Text(ct_frame, width=15, height=1)
        self.text_box_ct.pack(side="left", padx=5)
        self.text_box_ct.insert("1.0", "sample-name")
        # --------------------------------------------------------------

        # Function 2 => COMPUTED TOMOGRAPHY - ANGLE STEPS
        # ------------------
        angle_frame = tk.Frame(self.acquisition_frame)
        angle_frame.pack(pady=5, fill="x")
        #
        ct_angle = tk.Label(angle_frame, text="Angle range", width=20)
        ct_angle.pack(side="left")

        #
        self.text_box_angle_min = tk.Text(angle_frame, width=4, height=1)
        self.text_box_angle_min.pack(side="left", padx=6)
        self.text_box_angle_min.insert("1.0", "0")
        #
        ct_angle_min = tk.Label(angle_frame, text="MIN")
        ct_angle_min.pack(side="left", padx=(0, 10))

        #
        self.text_box_angle_max = tk.Text(angle_frame, width=4, height=1)
        self.text_box_angle_max.pack(side="left", padx=6)
        self.text_box_angle_max.insert("1.0", "180")
        #
        ct_angle_max = tk.Label(angle_frame, text="MAX")
        ct_angle_max.pack(side="left", padx=(0, 10))
        # --------------------------------------------------------------

        # Function 3 => COMPUTED TOMOGRAPHY - START
        # ------------------
        start_frame = tk.Frame(self.acquisition_frame)
        start_frame.pack(pady=5, fill="y")
        # -------------
        button = tk.Button(start_frame, text="Start", command=self._start_ct_acquisition)
        button.pack(side="left")
        # --------------------------------------------------------------

    # ---------------------- Actions for buttons ----------------------
    def _action_generic(self, spinbox, text_box, target_dir):
        # pass
        if not self._update_active:
            # Definiendo tiempo de exposicion
            exposure_time = self.exp_time_value.get().encode("ascii")

            # Leer el tiempo de exposición extendido
            status_exposure_time = pygigev.GevSetFeatureValueAsString(self.handle, b"ExtendedExposure", exposure_time)

            log.info(f"Setting Exposure Time: {status_exposure_time}")
            log.info(f"Exposure Time == {exposure_time}")

            numBuffers = int(spinbox.get())
            buffer_addresses = ((ctypes.c_void_p) * numBuffers)()

            # (Handle cases where image is larger than payload due to pixel unpacking)
            bufsize = self.payload_size.value
            bufsize_unpacked = (
                int(self.width_str.value)
                * int(self.height_str.value)
                * pygigev.GevGetPixelSizeInBytes(self.pixel_format_unpacked)
            )
            if bufsize_unpacked > bufsize:
                bufsize = bufsize_unpacked

            for bufIndex in range(numBuffers):
                temp = ((ctypes.c_char) * bufsize)()
                buffer_addresses[bufIndex] = ctypes.cast(temp, ctypes.c_void_p)
                log.info(f"buffer_addresses[{bufIndex}] = {hex(buffer_addresses[bufIndex])}")

            # Initialize a transfer (Asynchronous cycling)
            status = pygigev.GevInitializeTransfer(
                self.handle, pygigev.SynchronousNextEmpty, self.payload_size, numBuffers, buffer_addresses
            )
            log.info(f"Init transfer")

            # Grab images to fill the buffers
            numImages = numBuffers
            status = pygigev.GevStartTransfer(self.handle, numImages)
            log.info(f"Transfer already starts")

            # Read the images out
            gevbufPtr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()

            # index for files storage
            index = 0
            time0 = 5 * int(self.exp_time_value.get())
            print(type(time0), time0)
            tmout = (ctypes.c_uint32)(time0)
            log.info(f"TIME_OUT = {time0} miliseconds")

            for imgIndex in range(numImages):
                status = pygigev.GevWaitForNextFrame(self.handle, ctypes.byref(gevbufPtr), tmout.value)
                print(status)

                if status != 0:
                    continue

                # Check img data status
                gevbuf = gevbufPtr.contents

                if gevbuf.status != 0:
                    print("Img # ", imgIndex, " has status = ", gevbuf.status, "GEV_FRAME_STATUS_ERROR")
                    log.info(f"Img # {imgIndex} has status = {gevbuf.status} GEV_FRAME_STATUS_ERROR")
                    continue
                else:
                    print("Img # ", imgIndex, " has status = ", gevbuf.status, "GEV_FRAME_STATUS_RECVD")
                    log.info(f"Img # {imgIndex} has status = {gevbuf.status} GEV_FRAME_STATUS_RECVD")

                # Make a PIL image out of this frame
                im_addr = ctypes.cast(
                    gevbufPtr.contents.address, ctypes.POINTER(ctypes.c_ubyte * gevbufPtr.contents.recv_size)
                )

                image = np.frombuffer(im_addr.contents, dtype=np.uint16).reshape(
                    (int(self.height_str.value), int(self.width_str.value))
                )

                # saving image
                filename = f"{text_box.get("1.0", "end-1c")}{index:03d}.tif"
                filepath = target_dir.joinpath(filename)
                io.imsave(filepath, image)
                index += 1
                log.info(f"File {text_box.get("1.0", "end-1c")}{index:03d} already saved")

                # showing image
                width = int(int(self.width_str.value) * 0.15)
                height = int(int(self.height_str.value) * 0.15)
                pil_image = Image.fromarray(norm_img(image, out_range=(0, 255))).resize((width, height))
                tk_image = ImageTk.PhotoImage(pil_image)
                self.image_label.configure(image=tk_image)
                self.image_label.image = tk_image
                self.image_label.update_idletasks()

            # Free and Stop the transfer solo para modo synchrono
            for bufIndex in range(numBuffers):
                status = pygigev.GevReleaseFrameBuffer(self.handle, buffer_addresses[bufIndex])
                if status == 0:
                    log.info(f"Free buffer {buffer_addresses[bufIndex]} status = {status} successful")
                else:
                    log.info(f"Free buffer {buffer_addresses[bufIndex]} status = {status} error")

            status = pygigev.GevFreeTransfer(self.handle)
            if status == 0:
                messagebox.showinfo("Success", f"''{text_box.get("1.0", "end-1c")}'' images were successfully saved")
                log.info(msgs_.free_transfer)
            else:
                messagebox.showwarning(
                    "Warning",
                    "Free detector failed. but images were succesfully saved. Restart the app to avoid unexpected exit",
                )
                log.info(msgs_.free_transfer)

        else:
            messagebox.showwarning("Warning", "Press Stop button")

    def _action_flat(self):
        self._action_generic(spinbox=self.spinbox1, text_box=self.text_box1, target_dir=self._flat_dir)

    def _action_dark(self):
        self._action_generic(spinbox=self.spinbox2, text_box=self.text_box2, target_dir=self._dark_dir)

    def _action_bright(self):
        self._action_generic(spinbox=self.spinbox3, text_box=self.text_box3, target_dir=self._bright_dir)

    def _action_sample(self):
        samplepath = self._sample_dir.joinpath(self.text_box4.get("1.0", "end-1c"))
        samplepath.mkdir(parents=True, exist_ok=True)

        self._action_generic(spinbox=self.spinbox4, text_box=self.text_box4, target_dir=samplepath)

    def _start_ct_acquisition(self):
        # ---------------- Angle parameters ----------------
        min_angle = int(self.text_box_angle_min.get("1.0", "end-1c"))
        max_angle = int(self.text_box_angle_max.get("1.0", "end-1c"))
        interval = (
            self.text_box_angle_min.get("1.0", "end-1c")
            + "-"
            + self.text_box_angle_max.get("1.0", "end-1c")
        )
        proj2d_no = int(self.spinbox_ct.get())

        nanodeg = 1_000_000_000
        angle_step = int((max_angle - min_angle) / proj2d_no * nanodeg)

        # ---------------- Output directory ----------------
        ctpath = self._sample_dir.joinpath(
            f"ct_{self.text_box_ct.get('1.0', 'end-1c')}"
        )
        ctpath.mkdir(parents=True, exist_ok=True)

        # ---------------- Main acquisition ----------------
        if self._update_active:
            messagebox.showwarning("Warning", "Press Stop button")
            return

        # ==================================================
        # 1 CONFIGURE DETECTOR (NO TRANSFER ACTIVE)
        # ==================================================
        detector_control.configure_ext_trigger(
            self.handle,
            exposure_us=int(self.exp_time_value.get()),
            readout=b"FullFOV_2x2",   # cambia aquí si usas ROI pequeño
        )

        # ==================================================
        # 2 INIT MOTOR
        # ==================================================
        d_handle, channel, move_mode = motor_control.starting_motor("absolute")

        # ==================================================
        # 3 PREPARE BUFFERS
        # ==================================================
        numBuffers = proj2d_no
        buffer_addresses = ((ctypes.c_void_p) * numBuffers)()

        bufsize = self.payload_size.value
        bufsize_unpacked = (
            int(self.width_str.value)
            * int(self.height_str.value)
            * pygigev.GevGetPixelSizeInBytes(self.pixel_format_unpacked)
        )
        if bufsize_unpacked > bufsize:
            bufsize = bufsize_unpacked

        for i in range(numBuffers):
            temp = ((ctypes.c_char) * bufsize)()
            buffer_addresses[i] = ctypes.cast(temp, ctypes.c_void_p)

        # ==================================================
        # 4 INIT + START TRANSFER
        # ==================================================
        pygigev.GevInitializeTransfer(
            self.handle,
            pygigev.SynchronousNextEmpty,
            self.payload_size,
            numBuffers,
            buffer_addresses,
        )

        pygigev.GevStartTransfer(self.handle, numBuffers)

        gevbufPtr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()
        tmout = ctypes.c_uint32(5 * int(self.exp_time_value.get()))

        log.info("COMPUTED TOMOGRAPHY STARTED")

        # ==================================================
        # 5 ACQUISITION LOOP
        # ==================================================
        for imgIndex in range(numBuffers):

            # -------- Motor movement --------
            if imgIndex == 0:
                if min_angle == 0:
                    index = 0
                    angle_save = 0
                else:
                    index = motor_control.last_index(ctpath) + 1
                    angle_save = int(motor_control.last_position(ctpath))
                    motor_control.moving_motor(d_handle, channel, move_mode, angle_step)
                    angle_save += ctl.GetProperty_i64(d_handle, channel, ctl.PropertyKey.POSITION)
            else:
                motor_control.moving_motor(d_handle, channel, move_mode, angle_step)
                angle_save += int(
                    ctl.GetProperty_i64(d_handle, channel, ctl.PropertyKey.POSITION)
                )

            log.info(f"Img #{imgIndex} @ angle {angle_save}")

            # -------- Frame grab --------
            status = pygigev.GevWaitForNextFrame(
                self.handle, ctypes.byref(gevbufPtr), tmout.value
            )

            if status != 0:
                log.warning(f"Frame timeout / error: {status}")
                continue

            gevbuf = gevbufPtr.contents
            if gevbuf.status != 0:
                log.warning(f"Frame error status: {gevbuf.status}")
                continue

            # -------- Image extraction --------
            im_addr = ctypes.cast(
                gevbuf.address,
                ctypes.POINTER(ctypes.c_ubyte * gevbuf.recv_size),
            )

            image = np.frombuffer(im_addr.contents, dtype=np.uint16).reshape(
                (int(self.height_str.value), int(self.width_str.value))
            )

            # -------- Save image --------
            filename = f"img{index:05d}.tif"
            io.imsave(ctpath / filename, image)
            index += 1

            # -------- Display --------
            w = int(int(self.width_str.value) * 0.15)
            h = int(int(self.height_str.value) * 0.15)
            pil_image = Image.fromarray(
                norm_img(image, out_range=(0, 255))
            ).resize((w, h))

            tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image
            self.image_label.update_idletasks()

            motor_control.positions_storing(angle_save, ctpath, interval)

        # ==================================================
        # 6️⃣ CLEANUP
        # ==================================================
        motor_control.closing_motor(d_handle)
        pygigev.GevFreeTransfer(self.handle)

        messagebox.showinfo("Success", "CT projections successfully saved")
        log.info("COMPUTED TOMOGRAPHY STOPPED")



if __name__ == "__main__":
    project_directory_parser = my_parser()
    args = project_directory_parser.parse_args()
    project_directory = acq_directories.create(args.name)

    filename = "app.log"
    path_to_logfile = project_directory[0].joinpath(filename)

    # Setting of the log
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(path_to_logfile, mode="w"),  # Guardar logs en el archivo especificado
            logging.StreamHandler(),  # Mostrar logs en la consola (opcional)
        ],
    )

    root = tk.Tk()
    app = AcqViewerApp(root, project_directory)
    root.mainloop()
