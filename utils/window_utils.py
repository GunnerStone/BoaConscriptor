import win32gui
import win32con


def get_window_handle(target_window_prefix):
    """
    Retrieves the window handle of a window with a title that starts with the specified prefix.

    Parameters:
        target_window_prefix (str): The prefix of the window title to search for.

    Returns:
        int or None: The window handle (HWND) of the matching window, or None if no window is found.
    """
    window_handle = None

    def window_enum_callback(hwnd, _):
        nonlocal window_handle
        # print(hwnd, win32gui.GetWindowText(hwnd)) # uncomment to print all hwnd names (helpful for debugging)
        if win32gui.GetWindowText(hwnd).startswith(target_window_prefix):
            window_handle = hwnd

    win32gui.EnumWindows(window_enum_callback, None)
    return window_handle


def restore_window(hwnd):
    """
    Restores a minimized window identified by its window handle (HWND).

    Parameters:
        hwnd (int): The window handle (HWND) of the window to restore.
    """
    if win32gui.IsIconic(hwnd):  # Check if the window is minimized
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)


def focus_window(hwnd):
    """
    Brings a window identified by its window handle (HWND) to the foreground and gives it focus.
    If the window is minimized, it will be restored before being brought to the foreground.

    Parameters:
        hwnd (int): The window handle (HWND) of the window to focus.
    """
    restore_window(hwnd)
    win32gui.SetForegroundWindow(hwnd)


def get_window_size(hwnd):
    """
    Retrieves the size of the visible client area of a window identified by its window handle (HWND).

    Parameters:
        hwnd (int): The window handle (HWND) of the window to get the size of.

    Returns:
        tuple: A tuple (width, height) representing the dimensions of the visible client area of the window in pixels.
    """
    # Retrieve the dimensions of the client area
    client_rect = win32gui.GetClientRect(hwnd)
    width = client_rect[2]
    height = client_rect[3]
    return width, height


def set_window_size(hwnd, width, height):
    """
    Sets the size of a window identified by its window handle (HWND) while maintaining the original position.

    Parameters:
        hwnd (int): The window handle (HWND) of the window to set the size of.
        width (int): The desired width of the window in pixels.
        height (int): The desired height of the window in pixels.
    """
    # Get the current window position
    x, y, _, _ = win32gui.GetWindowRect(hwnd)

    # Set the new window size while maintaining the original position
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOP, x, y, width, height, win32con.SWP_SHOWWINDOW
    )


def get_window_position(hwnd):
    """
    Retrieves the position of the visible client area of a window identified by its window handle (HWND).

    Parameters:
        hwnd (int): The window handle (HWND) of the window to get the position of.

    Returns:
        tuple: A tuple (x, y) representing the coordinates of the top-left corner of the visible client area of the window in pixels.
    """
    # Retrieve the dimensions of the client area
    client_rect = win32gui.GetClientRect(hwnd)

    # Convert client coordinates to screen coordinates
    left, top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))

    return left, top


def set_window_position(hwnd, x=0, y=0):
    """
    Moves a window identified by its window handle (HWND) to the specified coordinates.

    Parameters:
        hwnd (int): The window handle (HWND) of the window to move.
        x (int): The desired x-coordinate (horizontal position) of the top-left corner of the window in pixels.
        y (int): The desired y-coordinate (vertical position) of the top-left corner of the window in pixels.
    """
    if is_taskbar_vertical():
        x += get_taskbar_width()
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOP, x, y, 0, 0, win32con.SWP_NOSIZE | win32con.SWP_NOZORDER
    )


def minimize_window(hwnd):
    """
    Minimizes a window identified by its window handle (HWND).

    Parameters:
        hwnd (int): The window handle (HWND) of the window to minimize.
    """
    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)


def maximize_window(hwnd):
    """
    Maximizes a window identified by its window handle (HWND).

    Parameters:
        hwnd (int): The window handle (HWND) of the window to maximize.
    """
    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)


def close_window(hwnd):
    """
    Closes a window identified by its window handle (HWND).

    Parameters:
        hwnd (int): The window handle (HWND) of the window to close.
    """
    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)


import win32gui


def get_taskbar_height():
    """
    Get the height of the taskbar.

    Returns:
        int: The height of the taskbar in pixels.
    """
    # Get the handle of the taskbar
    hwnd_taskbar = win32gui.FindWindow("Shell_TrayWnd", None)

    # Get the taskbar rectangle
    taskbar_rect = win32gui.GetWindowRect(hwnd_taskbar)

    # Calculate the taskbar height
    taskbar_height = taskbar_rect[3] - taskbar_rect[1]

    return taskbar_height


def get_taskbar_width():
    """
    Get the width of the taskbar.

    Returns:
        int: The width of the taskbar in pixels.
    """
    # Get the handle of the taskbar
    hwnd_taskbar = win32gui.FindWindow("Shell_TrayWnd", None)

    # Get the taskbar rectangle
    taskbar_rect = win32gui.GetWindowRect(hwnd_taskbar)

    # Calculate the taskbar width
    taskbar_width = taskbar_rect[2] - taskbar_rect[0]

    return taskbar_width


def is_taskbar_vertical():
    """
    Check if the taskbar is positioned vertically.

    Returns:
        bool: True if the taskbar is vertical, False if it is horizontal.
    """
    taskbar_width = get_taskbar_width()
    taskbar_height = get_taskbar_height()

    # If the taskbar is wider than its height, it is positioned horizontally
    return taskbar_width < taskbar_height
