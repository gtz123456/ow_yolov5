import cv2
import numpy as np
import pyautogui
import time
from grabscreen import grab_screen
import pynput
import win32con
import win32gui
import numpy as np  # 科学计算库
import cv2  # CV包
import torch  # PyTorch包
from grabscreen import grab_screen  # 实时截屏函数
from models.experimental import attempt_load  # 加载模型函数
from utils.general import non_max_suppression, scale_coords, xyxy2xywh  # NMS函数
from utils.datasets import letterbox  # 图像填充函数
import cv2
import keyboard
import mouse
import os

lock_on = False

def lockswitch():
    global lock_on
    if lock_on:
        lock_on = False
    else:
        lock_on = True

def f12():
    os._exit(0)

keyboard.add_hotkey('F4', lockswitch)
keyboard.add_hotkey('F12', f12)

'''
#cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
img = pyautogui.screenshot(region=[0,0,192,108]) # x,y,w,h
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2.imshow('csgo-detect', img)
cv2.waitKey(0)
'''

# ---------------------------设置参数------------------------
weights = r'./best.pt'  # 模型存放的位置
#weights = r'./aim-csgo2.pt'  # 模型存放的位置
window_size = (1920 // 2 - 32 * 7, 1080 // 2 - 32 * 6, 1920 // 2 + 32 * 7, 1080 // 2 + 32 * 6)  # 需要截屏的位置
re_x, re_y = 32 * 7 * 2, 32 * 6 * 2  # 1.cv2窗口大小 2.锁人范围大小
img_size = 320  # 输入到yolo5中的模型尺寸
conf_thres = 0.3  # NMS的置信度过滤
iou_thres = 0.2  # NMS的IOU阈值
view_img = False  # 是否观看目标检测结果
lock_mode = True  # 是否开启锁人
auto_fire = False  # 是否自动开火
plot = True  # 是否在原图画方框
# ----------------------------------------------------------

sim_input = mouse.sim_input()
# ---------------------------加载模型--------------------------------
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = attempt_load(weights, map_location=device)  # 加载FP32模型
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
half = device.type != 'cpu'
if half:
    model.half()  # to FP16 ---- 半精度训练
# -----------------------------------------------------------------

while True:

    # -----------------------图像预处理------------------------
    img0 = grab_screen(window_size)  # 获取屏幕图像
    img0 = cv2.resize(img0, (re_x, re_y))  # 可选，不加也没影响
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)  # opencv读取的是BGR格式，所以要转换

    # Padded resize ---- 图像填充
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert转换
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB ---- 灰度图转彩色图
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # FP16/32
    img /= 255.  # 归一化 0 - 255 to 0.0 - 1.0
    # 如果图片是3维(RGB) 就在前面添加一个维度1当中batch_size=1
    # 因为输入网络的图片需要是4维的 [batch_size, channel, w, h]
    if len(img.shape) == 3:
        img = img[None]  # 作用和 img = img.unsqueeze(0) 一样
    # -------------------------------------------------------

    
    # -----------------------模型预测-------------------------
    pred = model(img, augment=False)[0]  # 图像预测
    # print(pred)
    pred = non_max_suppression(pred, conf_thres, iou_thres)  # NMS
    # print(pred)   # 一张图片模型检测到多少个目标，pred输出的结果（张量）就有多少个数组（类和bbox）-----简单理解就是，一个整体里有多个目标，每个目标里有对应的类别和位置信息

    # Process predictions
    aims = []  # 存放多个目标，单个目标是个数组，里面存放[cls, x_c, y_c, w, h]
    for i, det in enumerate(pred):  # detections per image  ----  依次取出一个pred里的单个目标
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):  # 如果det不为零
            # Rescale boxes from img_size to im0 size   ----- 对单个目标的boxes（检测框）进行恢复操作，恢复到原图大小
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            # *xyxy 左上角坐标和右下角坐标     ————    conf 置信度    ————    cls 类别
            for *xyxy, conf, cls in reversed(det):
                # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format     ———— 对det转换成[cls, x_c, y_c, w, h]的5个元素的格式
                aim = (('%g ' * len(line)).rstrip() % line + '\n')  # 把tensor(0., device='cuda:0')转换成cls
                aim = aim.split(' ')  # 对空格进行分割，组成一个数组
                # print(aim)
                aims.append(aim)  # aim -- 存放目标信息[cls, x_c, y_c, w, h]

        # 锁人，并在原图上画框
        if len(aims):
            # 锁人  ----  lock会遍历aims里的每个目标
            # 自己选一个模式，同时开会检测到异常
            #if lock_mode:
                # lock1(aims, mouse, re_x, re_y, auto_fire)
                # lock2(aims, mouse, re_x, re_y, auto_fire)
                # lock3(aims, mouse, re_x, re_y, auto_fire)
                #lock4(aims, mouse, re_x, re_y, auto_fire)
            if plot:
                # 遍历aims里的每个目标，每个目标存放[cls, x_c, y_c, w, h]
                for i, det in enumerate(aims):
                    _, x_center, y_center, width, height = det
                    x_center, width = re_x * float(x_center), re_x * float(width)
                    y_center, height = re_y * float(y_center), re_y * float(height)
                    top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                    color = (0, 255, 0)  # RGB
                    cv2.rectangle(img0, top_left, bottom_right, color, thickness=3)  # 画框
                    if lock_on == True and i == 0:    
                        sim_input.move(int(x_center - re_x // 2),int(y_center - re_y // 2))
                        #lock_on = False
                        time.sleep(0.02)
                        sim_input.press()
                        sim_input.release()


    
    # -----------------------窗口设置-------------------------
    # 设置检测窗口的大小
    cv2.namedWindow('ow-detect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ow-detect', re_x // 3, re_y // 3)
    cv2.imshow('ow-detect', img0)
    # 设置检测窗口置顶
    hwnd = win32gui.FindWindow(None, 'ow-detect')
    CVRECT = cv2.getWindowImageRect('ow-detect')
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    # -----------------------模型预测-------------------------

    # waitKey()函数的功能是不断刷新图像，单位是ms
    # 如果waitKey(0)则会一直显示同一帧数
    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()  # 如果之前没有释放掉内存的操作的话destroyallWIndows会释放掉被那个变量占用的内存
        break
    