
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
import detector_control
import sys
import logging



logging.basicConfig(level=logging.INFO)

if len(sys.argv) != 2:
    print("Usage: python probe_trigger_mode.py <value>")
    sys.exit(1)

value = int(sys.argv[1])

# --- abrir detector ---
handle, *_ = detector_control.starting_detector()

# --- SET MUY TEMPRANO ---
val = ctypes.c_int(value)
ptr = ctypes.pointer(val)
size = ctypes.c_int(ctypes.sizeof(val))

status = pygigev.GevSetFeatureValue(
    handle,
    b"TriggerMode",
    size,
    ptr
)

print(f"TriggerMode = {value} -> status {status}")


# ==================================================
# 6 CLEANUP
# ==================================================
pygigev.GevFreeTransfer(handle)
pygigev.GevCloseCamera(ctypes.byref(handle))
pygigev.GevApiUninitialize()

# IMPORTANTE: salir inmediatamente
sys.exit(0)
