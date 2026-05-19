import ctypes
import multiprocessing
import sys
import tkinter as tk

from anystruct.main_application import Application


def _set_windows_dpi_awareness():
    if sys.platform != 'win32':
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass


def main(args=None):
    """Launch the ANYstructure Tkinter application."""
    if args is None:
        args = sys.argv[1:]

    multiprocessing.freeze_support()
    _set_windows_dpi_awareness()

    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.geometry(f'{width}x{height}')
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
