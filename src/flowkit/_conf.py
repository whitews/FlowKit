"""
Configuration & system settings used by FlowKit
"""
import sys
import platform

# Used to detect PyCharm's debugging mode to turn off multiprocessing for debugging with tests
get_trace = getattr(sys, 'gettrace', lambda: None)

# TODO: this seems to no longer work for recent versions of PyCharm
if get_trace() is None:
    debug = False
else:
    debug = True

    if 'coverage' in sys.modules.keys():
        # if running with coverage, don't use debug mode
        debug = False

_platform = platform.system().lower()

if _platform in ['linux', 'darwin']:
    # 'fork' is no longer favored, esp. on MacOS
    mp_context = 'forkserver'
else:
    # 'fork' not available on Windows
    mp_context = 'spawn'

# if for any reason multiprocessing is not available, turn it off
try:
    import multiprocessing as mp
    multi_proc = True
except ImportError:
    mp = None
    multi_proc = False
