import numpy as np
import ctypes
import sys
import logging as log

import pygigev
import msgs

msgs_ = msgs.messages()




# =======================================================================
# ============================   DETECTOR CONTROL   =====================
# Starting detector
def starting_detector():
    """Initializes and configures a GigE Vision camera connection.

    This function handles the initialization of the GigE Vision API, camera discovery,
    connection establishment and basic configuration. It includes:
    - API initialization
    - Camera detection and listing
    - Opening first available camera
    - Getting payload parameters and pixel format
    - Reading width and height information
    - Disabling turbo mode

    Returns:
        tuple: A tuple containing:
            - handle (c_void_p): Camera handle for subsequent operations
            - payload_size (c_uint64): Size of the image payload
            - pixel_format (c_uint32): Format of the pixels
            - pixel_format_unpacked (int): Unpacked pixel format
            - width_str (c_char array): Camera width as string
            - height_str (c_char array): Camera height as string

    Raises:
        RuntimeError: If no cameras are found or if there are errors during initialization
        Exception: For any other errors during the process

    Notes:
        The function automatically exits the program if critical errors occur during
        camera initialization or configuration.
    """
    try:
        # Initialize the API
        pygigev.GevApiInitialize()
        log.info(msgs_.open_detector)

        # Allocate a maximum number of camera info structures.
        maxCameras = 16
        numFound = (ctypes.c_uint32)(0)
        camera_info = (pygigev.GEV_CAMERA_INFO * maxCameras)()

        # Get the camera list
        status = pygigev.GevGetCameraList(camera_info, maxCameras, ctypes.byref(numFound))
        if status != 0:
            raise RuntimeError(f"Error {status} getting camera list")

        if numFound.value == 0:
            raise RuntimeError("No cameras found")

        # Select the first camera and open it.
        handle = (ctypes.c_void_p)()
        status = pygigev.GevOpenCamera(camera_info[0], pygigev.GevExclusiveMode, ctypes.byref(handle))

        if status != 0:
            raise RuntimeError(f"Error {status} opening camera")

        # Get the Width and Height (extra information)
        # Get the payload parameters
        payload_size = (ctypes.c_uint64)()
        pixel_format = (ctypes.c_uint32)()
        status = pygigev.GevGetPayloadParameters(handle, ctypes.byref(payload_size), ctypes.byref(pixel_format))
        log.info(f"Getting payload (payload size and pixel format) information: {status}")
        pixel_format_unpacked = pygigev.GevGetUnpackedPixelType(pixel_format)

        feature_strlen = (ctypes.c_int)(pygigev.MAX_GEVSTRING_LENGTH)
        unused = (ctypes.c_int)(0)
        if sys.version_info > (3, 0):
            width_name = b"Width"
            height_name = b"Height"
        else:
            raise RuntimeError(f"Error python version. Install python 3.x")


        width_str = ((ctypes.c_char) * feature_strlen.value)()
        height_str = ((ctypes.c_char) * feature_strlen.value)()
        status = pygigev.GevGetFeatureValueAsString(handle, width_name, unused, feature_strlen, width_str)
        log.info(f"Getting width information: {status}")
        status = pygigev.GevGetFeatureValueAsString(handle, height_name, ctypes.byref(unused), feature_strlen, height_str)
        log.info(f"Getting height information: {status}")


        # Defining parameters for TurboMode get dectivate
        int_turbomode = ctypes.c_int(0)
        value = ctypes.pointer(int_turbomode)
        size = ctypes.sizeof(int_turbomode)
        size_in_c_int_turbomode = ctypes.c_int(size)

        status = pygigev.GevSetFeatureValue(handle, b"transferTurboMode", size_in_c_int_turbomode, value)
        log.info(f"Deactivating TurboMode: {status}")

        return handle, payload_size, pixel_format, pixel_format_unpacked, width_str, height_str

    except Exception as e:
        log.error(f"Exception occurred: {e}")
        pygigev.GevApiUninitialize()
        sys.exit(1)

    # Initialize the API
    pygigev.GevApiInitialize()
    log.info(msgs_.open_detector)

    # Allocate a maximum number of camera info structures.
    maxCameras = 16
    numFound = (ctypes.c_uint32)(0)
    camera_info = (pygigev.GEV_CAMERA_INFO * maxCameras)()

    # Get the camera list
    status = pygigev.GevGetCameraList(camera_info, maxCameras, ctypes.byref(numFound))
    if status != 0:
        print("Error ", status, "getting camera list - exitting")
        log.error(f"Error {status}, getting camera list - exitting")
        sys.exit(1)

    if numFound.value == 0:
        print("No cameras found - exitting")
        log.error("No cameras found - exitting")
        quit()

    # Select the first camera and open it.
    handle = (ctypes.c_void_p)()
    status = pygigev.GevOpenCamera(camera_info[0], pygigev.GevExclusiveMode, ctypes.byref(handle))

    if status != 0:
        print("Error ", status, "opening camera - exiting")
        log.error(f"Error {status}, opening camera - exiting")
        quit()

    # --------------------------------------- Otras informaciones importantes ---------------------------------------
    # Get the Width and Height (extra information)
    # Get the payload parameters
    payload_size = (ctypes.c_uint64)()
    pixel_format = (ctypes.c_uint32)()
    status = pygigev.GevGetPayloadParameters(handle, ctypes.byref(payload_size), ctypes.byref(pixel_format))
    log.info(f"Getting payload (payload size and pixel format) information: {status}")
    pixel_format_unpacked = pygigev.GevGetUnpackedPixelType(pixel_format)

    feature_strlen = (ctypes.c_int)(pygigev.MAX_GEVSTRING_LENGTH)
    unused = (ctypes.c_int)(0)
    if sys.version_info > (3, 0):
        width_name = b"Width"
        height_name = b"Height"
    else:
        width_name = "Width"
        height_name = "Height"

    width_str = ((ctypes.c_char) * feature_strlen.value)()
    height_str = ((ctypes.c_char) * feature_strlen.value)()
    status = pygigev.GevGetFeatureValueAsString(handle, width_name, unused, feature_strlen, width_str)
    log.info(f"Getting width information: {status}")
    status = pygigev.GevGetFeatureValueAsString(handle, height_name, ctypes.byref(unused), feature_strlen, height_str)
    log.info(f"Getting height information: {status}")

    # ==============================================  Mi pequenho e insignificante aporte  ==============================================
    # Defining parameters for TurboMode get dectivate
    int_turbomode = ctypes.c_int(0)
    value = ctypes.pointer(int_turbomode)

    # Obtener el tamaño en bytes de lo que apunta 'value'
    size = ctypes.sizeof(int_turbomode)

    # Convertir el tamaño a un c_int
    size_in_c_int_turbomode = ctypes.c_int(size)

    status = pygigev.GevSetFeatureValue(handle, b"transferTurboMode", size_in_c_int_turbomode, value)
    log.info(f"Deactivating TurboMode: {status}")

    return handle, payload_size, pixel_format, pixel_format_unpacked, width_str, height_str


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

