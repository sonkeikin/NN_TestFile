import pyautogui
import pyperclip
import os
import time
import tools as tool
"""
time.sleep(2)
pyautogui.PAUSE=0.01

for i in range(100):
    pyperclip.copy("还不亲亲我")
    pyautogui.hotkey("Command","v")
    pyautogui.press("enter")
"""
msg = 'test'
tool.sendmail(msg)