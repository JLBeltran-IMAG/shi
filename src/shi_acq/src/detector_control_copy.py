"""
src.detector_control  detector_control.py
"""

import numpy as np
import ctypes
import sys
import logging as log
import pygigev
import msgs

msgs_ = msgs.messages()


log = log.getLogger(__name__)


def set_int_feature(handle, name: bytes, value: int):
    val = ctypes.c_int(value)
    ptr = ctypes.pointer(val)
    size = ctypes.sizeof(val)
    size_c = ctypes.c_int(size)

    status = pygigev.GevSetFeatureValue(
        handle,
        name,
        size_c,
        ptr
    )
    log.info(f"Set {name.decode()} = {value} -> status {status}")
    return status


def set_string_feature(handle, name: bytes, value: bytes):
    """
    Set a GenICam feature using string interface (GevSetFeatureValueAsString).
    This is REQUIRED for enum-like features such as ReadOutMode.
    """
    status = pygigev.GevSetFeatureValueAsString(
        handle,
        name,
        value
    )
    log.info(f"Set {name.decode()} = {value.decode()} -> status {status}")
    return status



def configure_safe_base(handle):
    """Estado base ultra estable"""
    set_int_feature(handle, b"transferTurboMode", 0)
    set_enum_feature(handle, b"TriggerMode", b"FreeRunning")
    set_int_feature(handle, b"NrExposedFrames", 1)
    set_int_feature(handle, b"NrOffsetFrames", 0)
    set_int_feature(handle, b"SumScheme", 0)
    set_int_feature(handle, b"ExtendedExposure", 500)


def configure_ext_trigger(handle, exposure_ms=20, readout=b"FullFOV_1x1"):
    """
    Configura el detector para trigger externo usando SOLO pygigev.
    """
    # --- Control: 1 trigger = 1 frame ---
    set_int_feature(handle, b"NrExposedFrames", 1)
    set_int_feature(handle, b"NrOffsetFrames", 0)
    set_int_feature(handle, b"SumScheme", 0)

    # --- Exposure (ms) ---
    set_int_feature(handle, b"ExtendedExposure", exposure_ms)


def dump_feature(handle, name):
    buf = ctypes.create_string_buffer(256)
    size = ctypes.c_int(256)
    unused = ctypes.c_int(0)

    status = pygigev.GevGetFeatureValueAsString(
        handle, name, unused, size, buf
    )

    print(f"{name.decode():25s} status={status} value={buf.value}")







# =======================================================================
# ============================   DETECTOR CONTROL   =====================
# Starting detector
def starting_detector():
    try:
        pygigev.GevApiInitialize()
        log.info(msgs_.open_detector)

        maxCameras = 16
        numFound = ctypes.c_uint32(0)
        camera_info = (pygigev.GEV_CAMERA_INFO * maxCameras)()

        status = pygigev.GevGetCameraList(camera_info, maxCameras, ctypes.byref(numFound))
        if status != 0 or numFound.value == 0:
            raise RuntimeError("No cameras found")

        handle = ctypes.c_void_p()
        status = pygigev.GevOpenCamera(camera_info[0], pygigev.GevExclusiveMode, ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"Error opening camera: {status}")
        
        # ---- SET TRIGGER MODE EARLY (CRITICAL) ----
        val = ctypes.c_int(0)  # 0 = FreeRunning
        ptr = ctypes.pointer(val)
        size = ctypes.c_int(ctypes.sizeof(val))

        status = pygigev.GevSetFeatureValue(
            handle,
            b"TriggerMode",
            size,
            ptr
        )
        log.info(f"Early TriggerMode=0 set -> status {status}")


        payload_size = ctypes.c_uint64()
        pixel_format = ctypes.c_uint32()
        pygigev.GevGetPayloadParameters(handle, ctypes.byref(payload_size), ctypes.byref(pixel_format))
        pixel_format_unpacked = pygigev.GevGetUnpackedPixelType(pixel_format)

        feature_strlen = ctypes.c_int(pygigev.MAX_GEVSTRING_LENGTH)
        unused = ctypes.c_int(0)

        width_str = (ctypes.c_char * feature_strlen.value)()
        height_str = (ctypes.c_char * feature_strlen.value)()

        pygigev.GevGetFeatureValueAsString(handle, b"Width", unused, feature_strlen, width_str)
        pygigev.GevGetFeatureValueAsString(handle, b"Height", unused, feature_strlen, height_str)

        return handle, payload_size, pixel_format, pixel_format_unpacked, width_str, height_str

    except Exception as e:
        log.error(f"Detector init failed: {e}")
        pygigev.GevApiUninitialize()
        sys.exit(1)


def taking_buffarray(handle, numBuffer, time_out=60000):
    """
    Captures and retrieves image data from a GigE Vision camera.

    This function starts image transfer from the camera, waits for the next frame,
    processes the received buffer data into a numpy array, and stops the transfer.

    Parameters:
    ----------
    handle : int
        Handle to the GigE Vision camera device
    numBuffer : int 
        Number of buffers to allocate for image capture
    time_out : int, optional
        Timeout value in milliseconds for frame capture (default is 60000)

    Returns:
    -------
    numpy.ndarray
        2D array containing the captured image data with dimensions (height, width)
        and dtype uint16

    Notes:
    -----
    The function uses the pygigev library to communicate with the camera and
    ctypes for memory management. The received image data is converted from
    the raw buffer to a numpy array preserving the original image dimensions.

    Prints status codes from GevStartTransfer, GevWaitForNextFrame and GevStopTransfer
    operations for debugging purposes.
    """
    # Grab images to fill the buffers
    status = pygigev.GevStartTransfer(handle, numBuffer)

    # Read the images out
    gevbufPtr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()

    tmout = (ctypes.c_uint32)(time_out)

    status = pygigev.GevWaitForNextFrame(handle, ctypes.byref(gevbufPtr), tmout.value)
    print(status)

    # Check img data status
    gevbuf = gevbufPtr.contents
    print(gevbuf.status)

    img_size = (gevbuf.h, gevbuf.w)
    img_addr = ctypes.cast(gevbuf.address, ctypes.POINTER(ctypes.c_ubyte * gevbuf.recv_size))

    img_array = np.frombuffer(img_addr.contents, dtype=np.uint16).reshape(img_size)

    status = pygigev.GevStopTransfer(handle)
    print(status)

    return img_array


# Stopping the detector
def stopping_transfer_and_detector(handle):
    """
    Stops the data transfer from the detector and closes the connection.

    This function performs the following operations:
    1. Aborts any ongoing transfer
    2. Frees transfer resources
    3. Closes the camera connection
    4. Uninitializes the GigE Vision API

    Parameters
    ----------
    handle : ctypes.c_void_p
        Handle to the GigE Vision camera/detector

    Returns
    -------
    None

    Notes
    -----
    The function logs various steps of the shutdown process using a logger instance.
    Status messages are both logged and printed to console for critical operations.
    """
    # Close camera
    status = pygigev.GevAbortTransfer(handle)
    log.info(msgs_.abort_transfer)
    status = pygigev.GevFreeTransfer(handle)
    log.info(msgs_.free_transfer)

    status = pygigev.GevCloseCamera(ctypes.byref(handle))
    if status == 0:
        print("Detector closed succesfully")
        log.info(msgs_.close_detector)

    else:
        log.error(f"Closing detector {status}")
        print("Closing detector error", status)

    # Uninitialize camera
    pygigev.GevApiUninitialize()
    log.info("Detector was uninitialize")

