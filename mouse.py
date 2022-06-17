from os import system
from comtypes.client import CreateObject
from random import randint

# 注册硬件
def Reg():
    system('E:\ow_yolov5_old\python64位调用幽灵键鼠\py64_com\GHOST.exe /RegServer')

# 注册
#Reg()

class sim_input:

    connected = False
    device = None

    def __init__(self):
        self.init()
    
    def init(self):
        self.device = CreateObject('GHOST.COM')
        open = self.device.IsDeviceConnected()
        if open != 1:
            print("未检测到幽灵键鼠")
        else:
            print("Connected")

    def move0(self,x,y):
        while x != 0 or y != 0:
            absx = abs(x)
            absy = abs(y)
            x0 = min(absx,randint(1,absx // 10 + 1))
            y0 = min(absy,randint(1,absy // 10 + 1)) #利用随机数实现鼠标平滑
            #x0 += randint(-1,2)
            #y0 += randint(-1,2)
            if x < 0:
                x0 = -x0
            if y < 0:
                y0 = -y0
            self.device.MoveMouseRelative(x0,y0)
            x -= x0
            y -= y0
            print(x,y)

    def move(self,x,y):
        while x != 0 or y != 0:
            absx = abs(x)
            absy = abs(y)
            x0 = min(absx,128)
            y0 = min(absy,128) #利用随机数实现鼠标平滑
            #x0 += randint(-1,2)
            #y0 += randint(-1,2)
            if x < 0:
                x0 = -x0
            if y < 0:
                y0 = -y0
            self.device.MoveMouseRelative(x0,y0)
            x -= x0
            y -= y0
            #print(x,y)

    def press(self):
        self.device.PressMouseButton(1)
    
    def release(self):
        self.device.ReleaseMouseButton(1)
        


