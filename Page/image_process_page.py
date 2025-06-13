#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     image_process_page.py
   @Description:   图像处理页面
-------------------------------------------------
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import sys
import pytesseract

# 设置 Tesseract 路径
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'

# 检查 Tesseract 安装
if not os.path.exists(tesseract_path):
    st.error(f"Tesseract 未在 {tesseract_path} 找到，请确保正确安装")
    st.stop()

# 检查语言数据文件
if not os.path.exists(os.path.join(tessdata_path, 'chi_sim.traineddata')):
    st.error(f"中文语言数据文件未在 {tessdata_path} 找到")
    st.error("请下载 chi_sim.traineddata 并放置在 tessdata 目录下")
    st.error("下载地址：https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata")
    st.stop()

# 设置 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# 设置环境变量
os.environ['TESSDATA_PREFIX'] = tessdata_path

# 设置页面标题
st.title("图像处理系统")

# 侧边栏配置
with st.sidebar:
    st.header("处理类型")
    process_type = st.selectbox(
        "选择处理类型",
        ["图像融合", "图像增强", "图像滤波", "图像分割", "OCR文字识别"]
    )

def image_fusion():
    """图像融合功能"""
    st.subheader("图像融合")
    # 上传第一张图片
    image1 = st.file_uploader("上传第一张图片", type=["jpg", "jpeg", "png"])
    if image1 is not None:
        image1 = Image.open(image1)
        image1 = np.array(image1)
        st.image(image1, caption="第一张图片", use_column_width=True)
    
    # 上传第二张图片
    image2 = st.file_uploader("上传第二张图片", type=["jpg", "jpeg", "png"])
    if image2 is not None:
        image2 = Image.open(image2)
        image2 = np.array(image2)
        st.image(image2, caption="第二张图片", use_column_width=True)
    
    # 融合参数
    alpha = st.slider("融合比例", 0.0, 1.0, 0.5)
    
    # 融合按钮
    if st.button("开始融合") and image1 is not None and image2 is not None:
        # 确保两张图片大小相同
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # 图像融合
        result = cv2.addWeighted(image1, alpha, image2, 1-alpha, 0)
        
        # 显示结果
        st.image(result, caption="融合结果", use_column_width=True)
        
        # 保存结果
        if st.button("保存结果"):
            # 创建结果目录
            os.makedirs("results", exist_ok=True)
            
            # 保存图片
            result_path = "results/fusion_result.jpg"
            cv2.imwrite(result_path, result)
            st.success(f"结果已保存到: {result_path}")
            
            # 添加下载按钮
            with open(result_path, "rb") as f:
                st.download_button(
                    label="下载融合结果",
                    data=f,
                    file_name="fusion_result.jpg",
                    mime="image/jpeg"
                )

def enhance_image():
    """图像增强功能"""
    st.subheader("图像增强")
    # 上传图片
    image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if image is not None:
        image = Image.open(image)
        image = np.array(image)
        st.image(image, caption="原图", use_column_width=True)
        
        # 增强参数
        brightness = st.slider("亮度", -100, 100, 0)
        contrast = st.slider("对比度", -100, 100, 0)
        
        # 增强按钮
        if st.button("开始增强"):
            # 图像增强
            result = cv2.convertScaleAbs(image, alpha=1+contrast/100, beta=brightness)
            
            # 显示结果
            st.image(result, caption="增强结果", use_column_width=True)
            
            # 保存结果
            if st.button("保存结果"):
                # 创建结果目录
                os.makedirs("results", exist_ok=True)
                
                # 保存图片
                result_path = "results/enhanced_result.jpg"
                cv2.imwrite(result_path, result)
                st.success(f"结果已保存到: {result_path}")
                
                # 添加下载按钮
                with open(result_path, "rb") as f:
                    st.download_button(
                        label="下载增强结果",
                        data=f,
                        file_name="enhanced_result.jpg",
                        mime="image/jpeg"
                    )

def filter_image():
    """图像滤波功能"""
    st.subheader("图像滤波")
    # 上传图片
    image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if image is not None:
        image = Image.open(image)
        image = np.array(image)
        st.image(image, caption="原图", use_column_width=True)
        
        # 滤波参数
        filter_type = st.selectbox(
            "选择滤波类型",
            ["高斯滤波", "中值滤波", "均值滤波"]
        )
        kernel_size = st.slider("核大小", 3, 15, 3, step=2)
        
        # 滤波按钮
        if st.button("开始滤波"):
            # 图像滤波
            if filter_type == "高斯滤波":
                result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            elif filter_type == "中值滤波":
                result = cv2.medianBlur(image, kernel_size)
            else:  # 均值滤波
                result = cv2.blur(image, (kernel_size, kernel_size))
            
            # 显示结果
            st.image(result, caption="滤波结果", use_column_width=True)
            
            # 保存结果
            if st.button("保存结果"):
                # 创建结果目录
                os.makedirs("results", exist_ok=True)
                
                # 保存图片
                result_path = "results/filtered_result.jpg"
                cv2.imwrite(result_path, result)
                st.success(f"结果已保存到: {result_path}")
                
                # 添加下载按钮
                with open(result_path, "rb") as f:
                    st.download_button(
                        label="下载滤波结果",
                        data=f,
                        file_name="filtered_result.jpg",
                        mime="image/jpeg"
                    )

def segment_image():
    """图像分割功能"""
    st.subheader("图像分割")
    # 上传图片
    image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if image is not None:
        image = Image.open(image)
        image = np.array(image)
        st.image(image, caption="原图", use_column_width=True)
        
        # 分割参数
        threshold = st.slider("阈值", 0, 255, 127)
        
        # 分割按钮
        if st.button("开始分割"):
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 图像分割
            _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # 显示结果
            st.image(result, caption="分割结果", use_column_width=True)
            
            # 保存结果
            if st.button("保存结果"):
                # 创建结果目录
                os.makedirs("results", exist_ok=True)
                
                # 保存图片
                result_path = "results/segmented_result.jpg"
                cv2.imwrite(result_path, result)
                st.success(f"结果已保存到: {result_path}")
                
                # 添加下载按钮
                with open(result_path, "rb") as f:
                    st.download_button(
                        label="下载分割结果",
                        data=f,
                        file_name="segmented_result.jpg",
                        mime="image/jpeg"
                    )

def ocr_recognition():
    """OCR文字识别功能"""
    st.subheader("OCR文字识别")
    # 上传图片
    image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    if image is not None:
        # 保存原始文件名
        original_filename = image.name
        image = Image.open(image)
        image = np.array(image)
        st.image(image, caption="原图", use_column_width=True)
        
        # OCR参数
        lang = st.selectbox(
            "选择识别语言",
            ["中文横排", "中文竖排", "英文", "中英文"]
        )
        # 语言映射
        lang_map = {
            "中文横排": "chi_sim",
            "中文竖排": "chi_sim_vert",
            "英文": "eng",
            "中英文": "chi_sim+eng"
        }
        
        # 预处理选项
        preprocess = st.checkbox("启用图像预处理", value=True)
        if preprocess:
            threshold = st.slider("二值化阈值", 0, 255, 127)
            denoise = st.checkbox("降噪", value=True)
        
        # 使用session_state存储识别结果
        if 'ocr_text' not in st.session_state:
            st.session_state.ocr_text = None
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = None
        if 'original_filename' not in st.session_state:
            st.session_state.original_filename = None
        
        # 识别按钮
        if st.button("开始识别"):
            with st.spinner("正在识别..."):
                # 图像预处理
                if preprocess:
                    # 转换为灰度图
                    if len(image.shape) == 3:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = image
                    
                    # 二值化
                    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                    
                    # 降噪
                    if denoise:
                        binary = cv2.medianBlur(binary, 3)
                    
                    processed_image = binary
                    st.image(processed_image, caption="预处理后的图片", use_column_width=True)
                else:
                    processed_image = image
                
                # 识别文字
                text = pytesseract.image_to_string(processed_image, lang=lang_map[lang])
                
                # 存储结果到session_state
                st.session_state.ocr_text = text
                st.session_state.processed_image = processed_image
                st.session_state.original_filename = original_filename
                
                # 显示识别结果
                st.subheader("识别结果")
                st.text_area("识别文本", text, height=200)
        
        # 保存结果按钮（移到识别按钮外部）
        if st.session_state.ocr_text is not None:
            if st.button("保存结果"):
                # 创建结果目录
                os.makedirs("results", exist_ok=True)
                
                # 保存文本
                text_path = f"results/ocr_result_{st.session_state.original_filename.split('.')[0]}.txt"
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(st.session_state.ocr_text)
                
                # 保存预处理后的图片
                if preprocess and st.session_state.processed_image is not None:
                    image_path = f"results/processed_{st.session_state.original_filename}"
                    cv2.imwrite(image_path, st.session_state.processed_image)
                
                st.success(f"结果已保存到: {text_path}")
                
                # 添加下载按钮
                with open(text_path, "rb") as f:
                    st.download_button(
                        label="下载识别结果",
                        data=f,
                        file_name=f"ocr_result_{st.session_state.original_filename.split('.')[0]}.txt",
                        mime="text/plain"
                    )

# 根据选择显示不同功能
if process_type == "图像融合":
    image_fusion()
elif process_type == "图像增强":
    enhance_image()
elif process_type == "图像滤波":
    filter_image()
elif process_type == "图像分割":
    segment_image()
elif process_type == "OCR文字识别":
    ocr_recognition() 