#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Description:   配置文件
-------------------------------------------------
"""
from pathlib import Path

# 模型目录配置
DETECTION_MODEL_DIR = Path("weights/detection")  # 目标检测模型目录
SEGMENT_MODEL_DIR = Path("weights/segment")      # 分割模型目录
POSE_MODEL_DIR = Path("weights/pose")            # 关键点检测模型目录

# 目标检测模型列表
DETECTION_MODEL_LIST = [
    "yolov8n.pt",  # 最小模型，速度最快
    "yolov8s.pt",  # 小型模型
    "yolov8m.pt",  # 中型模型
    "yolov8l.pt",  # 大型模型
    "yolov8x.pt"   # 超大型模型，精度最高
]

# 分割模型列表
SEGMENT_MODEL_LIST = [
    "yolov8n-seg.pt",  # 最小分割模型
    "yolov8s-seg.pt",  # 小型分割模型
    "yolov8m-seg.pt",  # 中型分割模型
    "yolov8l-seg.pt",  # 大型分割模型
    "yolov8x-seg.pt"   # 超大型分割模型
]

# 关键点检测模型列表
POSE_MODEL_LIST = [
    "yolov8n-pose.pt",  # 最小关键点检测模型
    "yolov8s-pose.pt",  # 小型关键点检测模型
    "yolov8m-pose.pt",  # 中型关键点检测模型
    "yolov8l-pose.pt",  # 大型关键点检测模型
    "yolov8x-pose.pt"   # 超大型关键点检测模型
]

# 数据源列表
SOURCES_LIST = ["图片", "视频", "摄像头"]  # 支持的数据源类型 