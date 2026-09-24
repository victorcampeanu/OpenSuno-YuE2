"""Small OS boundary; the server, workers and model pipeline stay shared."""
import os
from pathlib import Path
import signal
import subprocess
import time
import sysconfig

_dll_directories = []


def replace_file(source, target):
    """Retry brief Windows sharing violations while the UI reads progress."""
    for attempt in range(50):
        try:
            return source.replace(target)
        except PermissionError:
            if os.name != 'nt' or attempt == 49:
                raise
            time.sleep(.01)


def prepare_gpu_libraries():
    """Expose NVIDIA wheel DLLs to both Python and CUDA's native loader."""
    if os.name != 'nt' or _dll_directories:
        return
    package = Path(sysconfig.get_path('purelib')) / 'nvidia'
    directories = sorted({path.parent for path in package.rglob('*.dll')})
    for directory in directories:
        _dll_directories.append(os.add_dll_directory(str(directory)))
    os.environ['PATH'] = os.pathsep.join(map(str, directories)) + os.pathsep + os.environ.get('PATH', '')


def environment_python(root, analyze=False, environment=None):
    environment = environment or ('.transcribe-venv' if analyze else '.venv')
    return Path(root) / environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def stop_process(proc, force=False):
    if proc.poll() is not None:
        return
    try:
        if os.name == 'nt':
            # Include any audio helper processes owned by this worker.
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def watch_parent():
    # A Windows venv may launch a redirector between the server and worker.
    parent = int(os.environ.get('YUE2_PARENT_PID', os.getppid()))
    if os.name == 'nt':
        # Windows does not reparent orphaned processes, so polling getppid is
        # insufficient. Hold a handle to this exact parent until it exits.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, parent)  # SYNCHRONIZE
        if not handle:
            os._exit(1)
        result = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        kernel.CloseHandle(handle)
        os._exit(0 if result == 0 else 1)
    else:
        while os.getppid() == parent:
            time.sleep(2)
        os.killpg(os.getpgrp(), signal.SIGTERM)
