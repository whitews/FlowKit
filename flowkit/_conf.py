"""
Configuration & system settings used by FlowKit
"""
import sys
import platform

# Used to detect PyCharm's debugging mode to turn off multi-processing for debugging with tests
get_trace = getattr(sys, 'gettrace', lambda: None)

if get_trace() is None:
    debug = False
else:
    debug = True

_platform = platform.system().lower()

if _platform in ['linux', 'darwin']:
    mp_context = 'fork'
else:
    # fork not available on Windows
    mp_context = 'spawn'

# if for any reason multiprocessing is not available, turn it off
try:
    import multiprocessing as mp
    multi_proc = True
except ImportError:
    mp = None
    multi_proc = False