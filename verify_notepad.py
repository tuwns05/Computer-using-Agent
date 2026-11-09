"""Read Notepad's actual edit control to verify the terminal-agent smoke task."""
import ctypes
from ctypes import wintypes
import json
import psutil


def read_notepad():
    user = ctypes.windll.user32
    user.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user.SendMessageW.restype = ctypes.c_ssize_t
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    results = []

    def child(hwnd, unused):
        name = ctypes.create_unicode_buffer(256)
        user.GetClassNameW(hwnd, name, len(name))
        if name.value.lower().startswith('richedit') and user.IsWindowVisible(hwnd):
            length = user.SendMessageW(hwnd, 0xE, 0, 0)
            buffer = ctypes.create_unicode_buffer(min(max(length, 0), 65535)+1)
            user.SendMessageW(hwnd, 0xD, len(buffer), ctypes.addressof(buffer))
            results.append({'class': name.value, 'text': buffer.value})
        return True

    def window(hwnd, unused):
        if not user.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            if psutil.Process(pid.value).name().lower() == 'notepad.exe':
                user.EnumChildWindows(hwnd, callback_type(child), 0)
        except psutil.Error:
            pass
        return True

    user.EnumWindows(callback_type(window), 0)
    return results


if __name__ == '__main__':
    results = read_notepad()
    print(json.dumps(results, ensure_ascii=False))
    if not any('Xin chào Mage' in item['text'] for item in results):
        raise SystemExit('FAIL: expected text not found in Notepad')
    print('PASS: actual Notepad editor contains Xin chào Mage')
