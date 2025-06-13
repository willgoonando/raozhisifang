#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     yolo_page.py
   @Description:   YOLOv8目标检测页面
-------------------------------------------------
"""
from pathlib import Path
import streamlit as st
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# 设置页面标题
st.title("YOLOv8 目标检测系统")

# 创建侧边栏
st.sidebar.header("模型配置选择")

# 任务类型选择（检测/分割/关键点）
task_type = st.sidebar.selectbox(
    "任务类别选择",
    ["检测", "分割", "关键点"],
)

# 根据任务类型选择对应的模型
model_type = None
if task_type == "检测":
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.DETECTION_MODEL_LIST  # 从配置文件中获取检测模型列表
    )
elif task_type == "分割":
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.SEGMENT_MODEL_LIST    # 从配置文件中获取分割模型列表
    )
elif task_type == "关键点":
    model_type = st.sidebar.selectbox(
        "选择模型",
        config.POSE_MODEL_LIST       # 从配置文件中获取关键点检测模型列表
    )
else:
    st.error("Currently only 'Detection' function is implemented")

# 设置置信度阈值（30-100，默认50）
confidence = float(st.sidebar.slider(
    "置信度", 30, 100, 50)) / 100

# 根据任务类型和选择的模型，构建模型路径
model_path = ""
if model_type:
    if task_type == "检测":
        model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
    elif task_type == "分割":
        model_path = Path(config.SEGMENT_MODEL_DIR, str(model_type))
    elif task_type == "关键点":
        model_path = Path(config.POSE_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# 加载预训练模型
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# 数据源选择配置
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "选择来源",
    config.SOURCES_LIST  # 从配置文件中获取数据源列表
)

# 根据选择的数据源类型，调用相应的处理函数
source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # 图片
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # 视频
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # 摄像头
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented") 