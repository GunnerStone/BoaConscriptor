from utils.window_utils import get_window_position
from utils.window_utils import get_window_size
import pydirectinput
import pyautogui

import time

import numpy as np


def get_mouse_position():
    """
    Retrieves the current position of the mouse cursor.

    Returns:
        tuple: A tuple (x, y) representing the coordinates of the mouse cursor in pixels.
    """
    return pyautogui.position()


def click(x, y, hwnd, duration=0.0):
    """
    Performs a mouse click at the specified coordinates relative to the specified window.

    Parameters:
        x (int): The x-coordinate (horizontal position) relative to the window in pixels.
        y (int): The y-coordinate (vertical position) relative to the window in pixels.
        hwnd (int): The window handle (HWND) of the window in which to perform the click.
    """
    window_x, window_y = get_window_position(hwnd)
    pyautogui.click(x + window_x, y + window_y, duration=duration)


def press_key(key, interval=0.0):
    """
    Presses a key.

    Parameters:
        key (str): The key to press. It can be a single character or a special key code.
    """
    pydirectinput.press(key, interval=interval)


def type_string(string, interval=0.0):
    """
    Types a string by simulating key presses for each character.

    Parameters:
        string (str): The string to type.
    """
    pydirectinput.typewrite(string, interval=interval)


def press_key_combination(keys, interval=0.0):
    """
    Presses a combination of keys.

    Parameters:
        keys (list): A list of keys to press. Each key can be a single character or a special key code.
    """
    for key in keys:
        pydirectinput.keyDown(key, interval=interval)
    for key in keys:
        pydirectinput.keyUp(key, interval=interval)


def set_mouse_position(x, y, hwnd, duration=0.0):
    """
    Moves the mouse cursor to the specified coordinates relative to the specified window.

    Parameters:
        x (int): The x-coordinate (horizontal position) relative to the window in pixels.
        y (int): The y-coordinate (vertical position) relative to the window in pixels.
        hwnd (int): The window handle (HWND) of the window to move the mouse cursor in.
    """
    window_x, window_y = get_window_position(hwnd)
    pyautogui.moveTo(x + window_x, y + window_y, duration=duration)


def wind_mouse(
    hwnd,
    dest_x=None,
    dest_y=None,
    polling_rate=1000,
    G_0=9,
    W_0=3,
    M_0=15,
    D_0=12,
    move_mouse=lambda x, y: None,
):
    """
    WindMouse algorithm. Calls the move_mouse kwarg with each new step.
    Released under the terms of the GPLv3 license.
    G_0 - magnitude of the gravitational fornce
    W_0 - magnitude of the wind force fluctuations
    M_0 - maximum step size (velocity clip threshold)
    D_0 - distance where wind behavior changes from random to damped
    """
    start_x, start_y = get_mouse_position()

    if dest_x is None or dest_y is None:
        return

    # check to make sure dest_x and dest_y are within the window
    import win32gui

    window_width, window_height = get_window_size(hwnd)
    if dest_x > window_width or dest_y > window_height:
        print("Window size is {}".format((window_width, window_height)))
        print("Destination coords are at {}".format((dest_x, dest_y)))
        print("Destination mouse is outside of window, returning function")
        return

    # add hwnd x and y to dest_x and dest_y
    window_x, window_y = get_window_position(hwnd)
    dest_x += window_x
    dest_y += window_y

    sqrt3 = np.sqrt(3)
    sqrt5 = np.sqrt(5)

    current_x, current_y = start_x, start_y
    v_x = v_y = W_x = W_y = 0
    while np.hypot(dest_x - start_x, dest_y - start_y) >= 1:
        dist = np.hypot(dest_x - start_x, dest_y - start_y)
        # print("dist:", dist)
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
            W_y = W_y / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = np.random.random() * 3 + 3
            else:
                M_0 /= sqrt5
        v_x += W_x + G_0 * (dest_x - start_x) / dist
        v_y += W_y + G_0 * (dest_y - start_y) / dist
        v_mag = np.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0 / 2 + np.random.random() * M_0 / 2
            v_x = (v_x / v_mag) * v_clip
            v_y = (v_y / v_mag) * v_clip
        start_x += v_x
        start_y += v_y
        move_x = int(np.round(start_x))
        move_y = int(np.round(start_y))
        if current_x != move_x or current_y != move_y:
            # This should wait for the mouse polling interval
            pyautogui.moveTo(move_x, move_y, duration=0.0, _pause=False)
            time.sleep(1.0 / polling_rate)
            move_mouse(move_x, move_y)
            current_x, current_y = move_x, move_y
            pass
    return current_x, current_y
