#!/usr/bin/env python3


# -------#
# Source #
# -------#
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_dir))
# --------


import time
import ctypes
import numpy as np
import pygigev # type: ignore
import detector_control_copy as detector_control # type: ignore
import logging

import msgs # type: ignore

msgs_ = msgs.messages()


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("trigger-test")


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
    set_int_feature(handle, b"transferTurboMode", 0)
    set_string_feature(handle, b"TriggerMode", b"FreeRunning")
    set_int_feature(handle, b"NrExposedFrames", 1)
    set_int_feature(handle, b"NrOffsetFrames", 0)
    set_int_feature(handle, b"SumScheme", 0)
    set_int_feature(handle, b"ExtendedExposure", 500)

def configure_ext_trigger(handle, exposure_ms=20, readout=b"FullFOV_1x1"):
    configure_safe_base(handle)
    set_string_feature(handle, b"ReadoutMode", readout)
    set_string_feature(handle, b"TriggerMode", b"ExtTrigger")
    set_int_feature(handle, b"NrExposedFrames", 1)
    set_int_feature(handle, b"NrOffsetFrames", 0)
    set_int_feature(handle, b"SumScheme", 0)
    set_int_feature(handle, b"ExtendedExposure", int(exposure_ms))

def get_int_feature(handle, name: bytes) -> int:
    feature_type = ctypes.c_int()
    v = ctypes.c_int64()

    status = pygigev.GevGetFeatureValue(
        handle,
        name,
        ctypes.byref(feature_type),
        ctypes.sizeof(v),
        ctypes.byref(v),
    )
    if status != 0:
        raise RuntimeError(f"Failed to get {name!r}")

    return int(v.value)

def get_string_feature(handle, name: bytes) -> str:
    feature_type = ctypes.c_int()
    buf = ctypes.create_string_buffer(256)

    status = pygigev.GevGetFeatureValueAsString(
        handle,
        name,
        ctypes.byref(feature_type),
        len(buf),
        buf,
    )
    if status != 0:
        raise RuntimeError(f"Failed to get {name!r}")
    else:
        print(f"{name.decode()}: {buf.value.decode()}")

    return buf.value.decode()

def starting_detector():
    try:
        pygigev.GevApiInitialize()
        log.info(msgs_.open_detector)

        max_cameras = 16
        num_found = ctypes.c_uint32(0)
        camera_info = (pygigev.GEV_CAMERA_INFO * max_cameras)()

        status = pygigev.GevGetCameraList(
            camera_info,
            max_cameras,
            ctypes.byref(num_found)
        )
        if status != 0 or num_found.value == 0:
            raise RuntimeError("No cameras found")

        handle = ctypes.c_void_p()
        status = pygigev.GevOpenCamera(
            camera_info[0],
            pygigev.GevExclusiveMode,
            ctypes.byref(handle)
        )
        if status != 0:
            raise RuntimeError(f"Error opening camera: {status}")

        # --------------------------------------------------
        # Payload & pixel format (READ ONLY)
        # --------------------------------------------------
        payload_size = ctypes.c_uint64()
        pixel_format = ctypes.c_uint32()

        pygigev.GevGetPayloadParameters(
            handle,
            ctypes.byref(payload_size),
            ctypes.byref(pixel_format)
        )

        pixel_format_unpacked = pygigev.GevGetUnpackedPixelType(pixel_format)

        # --------------------------------------------------
        # Image size (read as INT, not string)
        # --------------------------------------------------
        width = get_int_feature(handle, b"Width")
        height = get_int_feature(handle, b"Height")

        return (
            handle,
            payload_size,
            pixel_format,
            pixel_format_unpacked,
            width,
            height,
        )

    except Exception as e:
        log.error(f"Detector init failed: {e}")
        pygigev.GevApiUninitialize()
        sys.exit(1)

def stopping_transfer_and_detector(handle):
    """
    Safely stop acquisition, release transfer resources,
    close the camera, and uninitialize the GigE Vision API.
    """

    if not handle or handle.value is None:
        log.warning("Invalid camera handle, skipping shutdown")
        pygigev.GevApiUninitialize()
        return

    # --------------------------------------------------
    # 1. Stop / abort transfer (best effort)
    # --------------------------------------------------
    try:
        status = pygigev.GevAbortTransfer(handle)
        if status == 0:
            log.info("Transfer aborted")
        else:
            log.debug(f"GevAbortTransfer status: {status}")
    except Exception as e:
        log.debug(f"AbortTransfer exception ignored: {e}")

    # --------------------------------------------------
    # 2. Free transfer resources
    # --------------------------------------------------
    try:
        status = pygigev.GevFreeTransfer(handle)
        if status == 0:
            log.info("Transfer resources freed")
        else:
            log.debug(f"GevFreeTransfer status: {status}")
    except Exception as e:
        log.debug(f"FreeTransfer exception ignored: {e}")

    # --------------------------------------------------
    # 3. Close camera
    # --------------------------------------------------
    try:
        status = pygigev.GevCloseCamera(ctypes.byref(handle))
        if status == 0:
            log.info("Detector closed successfully")
        else:
            log.warning(f"GevCloseCamera returned status {status}")
    except Exception as e:
        log.error(f"Exception while closing camera: {e}")

    # --------------------------------------------------
    # 4. Uninitialize API (always)
    # --------------------------------------------------
    try:
        pygigev.GevApiUninitialize()
        log.info("GigE Vision API uninitialized")
    except Exception as e:
        log.error(f"Failed to uninitialize API: {e}")


def main():

    # ==================================================
    # 1. OPEN DETECTOR
    # ==================================================
    (
        handle,
        payload_size,
        pixel_format,
        pixel_format_unpacked,
        width,
        height,
    ) = starting_detector()

    log.info(f"Detector opened | {width} x {height}")

    # ==================================================
    # 2. CONFIGURE EXTERNAL TRIGGER
    # ==================================================
    configure_ext_trigger(
        handle,
        exposure_ms=50,
        readout=b"FullFOV_1x1",
    )

    log.info("Detector configured for external trigger")


    # ==================================================
    # 3. PREPARE BUFFERS
    # ==================================================
    num_buffers = 16
    buffer_addresses = (ctypes.c_void_p * num_buffers)()

    bufsize = payload_size.value
    bufsize_unpacked = (
        int(width)
        * int(height)
        * pygigev.GevGetPixelSizeInBytes(pixel_format_unpacked)
    )

    if bufsize_unpacked > bufsize:
        bufsize = bufsize_unpacked

    buffers = []
    for i in range(num_buffers):
        buf = (ctypes.c_char * bufsize)()
        buffers.append(buf)  # mantener referencia viva
        buffer_addresses[i] = ctypes.cast(buf, ctypes.c_void_p)

    log.info(
        f"Buffers ready | {num_buffers=} | {bufsize/1024/1024:.2f} MB each"
    )

    # ==================================================
    # 4. INIT + START TRANSFER
    # ==================================================
    pygigev.GevInitializeTransfer(
        handle,
        pygigev.SynchronousNextEmpty,
        payload_size,
        num_buffers,
        buffer_addresses,
    )

    pygigev.GevStartTransfer(handle, num_buffers)

    log.info("Waiting for external triggers (Ctrl+C to stop)")

    # ==================================================
    # 5. ACQUISITION LOOP
    # ==================================================
    gevbuf_ptr = ctypes.POINTER(pygigev.GEV_BUFFER_OBJECT)()
    last_time = None
    frame_count = 0

    try:
        while True:
            timeout_ms = 2000

            status = pygigev.GevWaitForNextFrame(
                handle,
                ctypes.byref(gevbuf_ptr),
                timeout_ms,
            )

            if status != 0:
                continue

            gevbuf = gevbuf_ptr.contents
            if gevbuf.status != 0:
                log.warning(f"Frame error status: {gevbuf.status}")
                continue

            now = time.perf_counter()

            if last_time is None:
                log.info("First frame received")
            else:
                dt = now - last_time
                log.info(
                    f"Frame {frame_count:05d} | "
                    f"Δt = {dt*1000:7.2f} ms | "
                    f"FPS ≈ {1/dt:7.2f}"
                )

            last_time = now
            frame_count += 1

    except KeyboardInterrupt:
        log.info("Stopping acquisition...")

    finally:
        # ==================================================
        # 6. CLEANUP
        # ==================================================
        stopping_transfer_and_detector(handle)
        log.info(f"Total frames received: {frame_count}")


if __name__ == "__main__":
    main()

