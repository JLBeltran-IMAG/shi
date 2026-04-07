# -------#
# Source #
# -------#
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_dir))
# --------

import ctypes
import pygigev
import detector_control_copy as detector_control
import logging

logging.basicConfig(level=logging.INFO)

# Abrir detector
handle, *_ = detector_control.starting_detector()

# --- Volver a FreeRunning ---
val = ctypes.c_int(0)   # 0 = FreeRunning
ptr = ctypes.pointer(val)
size = ctypes.c_int(ctypes.sizeof(val))

# -----------------------------
# 2) Readout mode
# -----------------------------
# FullFOV_1x1, FullFOV_2x2, FullFOV_1x2
detector_control.set_string_feature(handle, b"ReadOutMode", b"FullFOV_1x1")

# Or the integer values 3, 4, 5 respectively
detector_control.set_int_feature(handle, b"ReadOutMode", 3)


# -----------------------------
# 2) Trigger mode
# -----------------------------
# FreeRunning, ExtTrigger, Snapshot, TimedSnap, TrigSequence
detector_control.set_string_feature(handle, b"TriggerMode", b"ExtTrigger")

# Or the integer values 0, 1, 2, 3, 4 respectively
detector_control.set_int_feature(handle, b"TriggerMode", 1)




# Cerrar correctamente
pygigev.GevAbortTransfer(handle)
pygigev.GevFreeTransfer(handle)
pygigev.GevCloseCamera(ctypes.byref(handle))
pygigev.GevApiUninitialize()
