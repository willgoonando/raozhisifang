import streamlit as st

st.title("圆形检测")
import cv2
import numpy as np

files = st.file_uploader('图片上传', type=["jpg", "png", "jpeg"])

# 将页面拆分为两个布局
col1, col2 = st.columns(2)

if files is not None:
    values = files.getvalue()
    # 转换为opencv可以识别的格式
    cv2_img = cv2.imdecode(np.frombuffer(values, np.uint8), cv2.IMREAD_COLOR)

    # 将左边布局显示原始图像
    with col1:
        st.image(cv2_img, '原始图像', channels="BGR")

    def method(img):
        # 不需要再次读取图像，直接使用传入的img
        im = img.copy()  # 复制一份，避免修改原图
        
        # 转换为灰度图
        im_gary = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # 检测圆形
        circles = cv2.HoughCircles(im_gary, cv2.HOUGH_GRADIENT, 20, 
                                  param1=200, param2=100, 
                                  minDist=100, maxRadius=500)
        
        if circles is not None:
            # 绘制检测到的圆和圆心
            for x, y, r in circles[0]:
                # 绘制圆的轮廓
                cv2.circle(im, (int(x), int(y)), int(r), (0, 255, 0), 2)
                # 绘制圆心
                cv2.circle(im, (int(x), int(y)), 2, (255, 255, 0), 2)
                
        return im

    # 在右侧显示检测结果
    with col2:
        result_img = method(cv2_img)
        st.image(result_img, '检测结果', channels="BGR")