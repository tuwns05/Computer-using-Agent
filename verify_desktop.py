"""Exercise mouse and Unicode keyboard input against a disposable test window."""
import tkinter as tk
import mage


def main():
    desktop = mage.Desktop()
    root = tk.Tk()
    root.title('Mage input verification')
    root.geometry('500x150+200+200')
    field = tk.Entry(root, font=('Segoe UI', 16))
    field.pack(fill='x', padx=20, pady=30)
    try:
        root.update()
        root.lift()
        root.focus_force()
        field.focus_set()
        root.update()
        desktop.screenshot()
        x = (field.winfo_rootx()+20)*1000/desktop.width
        y = (field.winfo_rooty()+10)*1000/desktop.height
        desktop.execute({'action': 'click', 'coordinate': [x, y]})
        root.update()
        desktop.type_text('Xin chào Mage 🪄')
        root.update()
        assert field.get() == 'Xin chào Mage 🪄', repr(field.get())
        desktop.keypress(['ctrl', 'a'])
        desktop.keypress(['backspace'])
        root.update()
        assert field.get() == '', repr(field.get())
        print('PASS: screenshot, click, Unicode input, Ctrl+A, Backspace')
    finally:
        root.destroy()


if __name__ == '__main__':
    main()
