import pyautogui
import cv2
import keyboard

a = 280

def screenshot():
    global a
    img = pyautogui.screenshot(region=[0,0,1920,1080]) # x,y,w,h
    img.save(str(a) + '.png')
    a += 1
    #img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
keyboard.add_hotkey('F4', screenshot)
keyboard.wait('F12')