import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import re
import time
import os
import serial

# 固定尺寸
def resizeImg(image, height=480):#height调成图像高度
    h, w = image.shape[:2]
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img

# 边缘检测
def getCanny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 240, apertureSize=3)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(edges, kernel, iterations=2)
    return binary

def findMaxContour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:  # 如果 contours 列表为空
        return None, 0.0  # 返回空的轮廓和面积为 0
    max_area = 0.0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour, max_area

def getBoxPoint(contour):
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx

def adaPoint(box, pro):
    box_pro = box
    if pro != 1.0:
        box_pro = box/pro
    box_pro = np.trunc(box_pro)
    return box_pro

def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
# 计算长宽
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))
# 透视变换
def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0],
                         [w - 1, 0],
                         [w - 1, h - 1],
                         [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(np.array(box, dtype='float32'), dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    Minv = cv2.getPerspectiveTransform(dst_rect, np.array(box, dtype='float32'))
    
    return warped, Minv



def ocr_with_position(image):
    # 初始化OCR模型
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 只需要运行一次以下载并加载模型到内存中
    
    # 执行OCR
    result = ocr.ocr(image, cls=True)
    
    # 提取文本框位置信息
    text_boxes = []
    for res in result:
        for line in res:
            text_box = line[0]  # 文本框坐标 (x1, y1, x2, y2)
            text_boxes.append(text_box)
    
    # 返回OCR内容和文本框位置信息的列表
    return result, text_boxes

def check_math_errors(ocr_results):
    # 用于存储计算错误的数学算式
    error_expressions = []
    unrecognized=[] 
    # 遍历 OCR 结果列表
    for line in ocr_results:
        # 获取文本内容和置信度
        for l in line:
            text=l[1][0]
            box=l[0]
            print(l)
            # 使用正则表达式检查是否为数学表达式
            if re.match(r'^\d+(\.\d+)?[+\-*/×x÷]\d+(\.\d+)?=\d+(\.\d+)?$', text):
                # 分割算式为左右两部分
                text = text.replace('×', '*').replace('÷', '/').replace('x', '*')
                left, right = text.split('=')
                # 如果等号两边的结果不相等，则打印错误
                if eval(left) != eval(right):
               
                    error_expressions.append(box)
            else:
                unrecognized.append(box)
    return error_expressions,unrecognized


def visualize_ocr(image, text_boxes):
    # 标出文本框位置
    image_with_boxes = draw_ocr(image, text_boxes)
    image_pil = Image.fromarray(image_with_boxes)
    image_pil.show()
    # 返回标出文本框位置后的图像
    return image_with_boxes

# 点的逆透视变换函数
def transformPoint(pt, Minv):
    point = np.array([[[pt[0], pt[1]]]], dtype='float32')
    transformed_point = cv2.perspectiveTransform(point, Minv)
    return transformed_point[0][0]

# 多个点的逆透视变换函数
def transformBox(box, Minv):
    box = np.array(box, dtype='float32')
    transformed_box = cv2.perspectiveTransform(np.array([box]), Minv)
    return transformed_box[0]


def adjust_text_boxes_for_perspective(text_boxes, box, Minv):
    # 调整文本框位置信息
    w, h = pointDistance(box[0], box[1]), pointDistance(box[1], box[2])
    adjusted_text_boxes = []
    for text_box in text_boxes:
        tsbox=transformBox(text_box,Minv)
        real_corners = []
        for corner in tsbox:
            x, y = corner
            real_x = x + box[0][0]
            real_y = y + box[0][1]
            real_corners.append([real_x, real_y])
        
        adjusted_text_boxes.append(real_corners)
    
    return adjusted_text_boxes

# 已知物体的实际宽度
known_width = 0.2  # 单位：米

def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width


# 计算相机的焦距
def find_focal_length(known_distance, known_width, per_width):
    return (per_width * known_distance) / known_width

def detect(image):   
    try:
        ratio = 480 / image.shape[0]
        img = resizeImg(image)

        binary_img = getCanny(img)
        max_contour, max_area = findMaxContour(binary_img)

        if max_contour is None:
            raise ValueError("No contours found")

        boxes1 = getBoxPoint(max_contour)
        boxes2 = adaPoint(boxes1, ratio)
        boxes = orderPoints(boxes2)  # 得到四个角点坐标

        if boxes is None:
            raise ValueError("No contours found")

        # 透视变化
        warped, Minv = warpImage(image, boxes)  # 透视图和逆变换矩阵
        # OCR得到
        result, text_boxes = ocr_with_position(image)  # 结果（包含文本框位置）和文本框位置

        if result is None or text_boxes is None:
            raise ValueError("OCR failed to recognize text")

        errors, unrecognized = check_math_errors(result)  # 错误和未识别的框坐标列表 注意 是在透视变换后的图中
        return errors, unrecognized, boxes

    except Exception as e:
        print(f"Detection error: {e}")
        return [], [],[]

def save_frame(frame,save_dir="/home/hjx/Project/Project/photos",max_images=10):
    timestamp=time.strftime("%Y%m%d-%H%M%S")
    filename=os.path.join(save_dir,f"frame_{timestamp}.jpg")
    cv2.imwrite(filename,frame)

    images=sorted(os.listdir(save_dir))
    if len(images) > max_images :
        os.remove(os.path.join(save_dir,images[0]))
    

def open_camera(device_index=0):
    """
    打开摄像机外设并实时显示视频画面。
    
    参数：
    device_index: int, 可选，默认值为0，表示使用第一个摄像头设备。
    """

    ser=serial.Serial('/dev/ttyS0',115200)
    if ser.isOpen==False :
        ser.open()
    ser.write(b"Raspberry pi is ready\r\n")


    # 打开摄像机
    cap = cv2.VideoCapture(device_index)
    
    # 检查摄像头是否打开成功
    if not cap.isOpened():
        print(f"无法打开摄像头（设备索引: {device_index}）")
        return

    # 获取摄像头的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {width}x{height}")
   

    timeF=10
    c=0;
    flag=0

    have_receive = 0
    last_size = 0


    overlay=np.zeros((height,width,3),dtype='uint8')
    while True:
        #time.sleep(interval)

        # 读取摄像头帧
        ret, frame = cap.read()
        
        # 检查是否成功读取帧
        if not ret:
            print("无法接收帧（可能是摄像头断开了连接）")
            break
        
        #十帧取一帧
        if c%timeF==0 and flag==0  :

            # 调用检测函数，获取框坐标列表
            err_boxes, unr_boxes,boxes = detect(frame)
            
            # 创建一个拷贝的帧用于绘制
            draw_frame = frame.copy()
     
                    # 绘制 err 框
            if err_boxes:
                for box in err_boxes:
                    if len(box) == 4:
                        pts = np.array([box], np.int32)  # 将四个点的坐标放入一个数组中
                        pts = pts.reshape((-1, 1, 2))    # 将数组重塑为适合 cv2.polylines 的形状
                        cv2.polylines(draw_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)  # 画多边形

        # 检查 boxes 是否包含四个点，并绘制多边形
            if len(boxes) == 4:
                pts = np.array([boxes], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(draw_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # 绘制 unr_boxes 多边形
            if unr_boxes:
                for box in unr_boxes:
                    if len(box) == 4:
                        pts = np.array([box], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(draw_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)


            # 将绘制的帧赋给 overlay
            overlay = draw_frame    
            # 将 overlay 显示在窗口中
            cv2.imshow('Camera', overlay)
            
            save_frame(overlay)
            flag=1
            ser.write(b'begin\r\n')

        c=c+1

        
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        size=ser.inWaiting()
        if size!=0:
            if have_receive == 0:
                time.sleep(0.5)
                have_receive = 1
                last_size = size
            else :
                if(size != last_size):
                    time.sleep(0.5)
                    last_size = size
                else:
                    response=ser.read(size)
                    print(response)
                    if response==b'over\r\n':
                        flag=0
                    ser.flushInput()
                    last_size = 0
                    have_receive = 0
                    time.sleep(0.5)


# 调用函数打开摄像机
open_camera(device_index=0)


    





