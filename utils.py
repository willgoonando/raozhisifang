#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Description:   工具函数
-------------------------------------------------
"""
import cv2
import torch
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import numpy as np
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
from pathlib import Path
import time

def load_model(model_path):
    """
    加载YOLO模型
    :param model_path: 模型路径
    :return: YOLO模型实例
    """
    return YOLO(model_path)

def save_detection_results(results, image_name):
    """保存检测结果到JSON文件
    
    Args:
        results: 检测结果
        image_name: 图片名称
    """
    # 创建results目录
    os.makedirs("results", exist_ok=True)
    
    # 准备结果数据
    result_data = {
        "image_name": image_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detections": []
    }
    
    # 处理检测结果
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        result_data["detections"].append({
            "class": results[0].names[cls],
            "confidence": conf,
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })
    
    # 保存为JSON
    json_path = f"results/{image_name.split('.')[0]}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    return json_path

def visualize_detection_results(results, conf):
    """可视化检测结果
    Args:
        results: 检测结果
        conf: 置信度阈值
    """
    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    
    # 获取检测框
    boxes = results[0].boxes
    
    # 第一列：基本统计信息
    with col1:
        st.subheader("基本统计")
        st.write(f"检测到的目标数量: {len(boxes)}")
        if len(boxes) > 0:
            confidences = [float(box.conf[0]) for box in boxes]
            st.write(f"置信度范围: {min(confidences):.2f} - {max(confidences):.2f}")
    
    # 第二列：类别分布饼图
    with col2:
        if len(boxes) > 0:
            st.subheader("类别分布")
            # 统计每个类别的数量
            class_counts = {}
            for box in boxes:
                cls = int(box.cls[0])
                class_name = results[0].names[cls]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # 创建饼图
            fig = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title="类别分布"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 第三列：置信度分布直方图
    with col3:
        if len(boxes) > 0:
            st.subheader("置信度分布")
            # 创建置信度直方图
            confidences = [float(box.conf[0]) for box in boxes]
            fig = px.histogram(
                x=confidences,
                title="置信度分布",
                labels={"x": "置信度", "y": "数量"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 显示详细结果表格
    if len(boxes) > 0:
        st.subheader("详细检测结果")
        # 创建结果表格
        results_data = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            results_data.append({
                "类别": results[0].names[cls],
                "置信度": f"{conf:.2f}",
                "位置": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            })
        
        # 显示表格
        st.dataframe(pd.DataFrame(results_data))

def process_batch_images(folder_path, conf, model):
    """批量处理文件夹中的图片
    
    Args:
        folder_path: 图片文件夹路径
        conf: 置信度阈值
        model: YOLO模型实例
    """
    # 创建结果目录
    results_dir = os.path.join(folder_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    
    if not image_files:
        st.warning("文件夹中没有找到图片文件")
        return
    
    # 创建进度条
    progress_bar = st.progress(0)
    
    # 创建结果汇总
    all_results = []
    
    # 处理每张图片
    for idx, image_file in enumerate(image_files):
        # 更新进度
        progress_bar.progress((idx + 1) / len(image_files))
        
        # 读取图片
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        
        # 进行预测
        results = model.predict(image, conf=conf)
        
        # 保存检测结果
        result_image = results[0].plot()[:, :, ::-1]
        result_path = os.path.join(results_dir, f"detected_{image_file}")
        cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # 保存JSON结果
        json_path = save_detection_results(results, image_file)
        
        # 添加到汇总结果
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            all_results.append({
                "image": image_file,
                "class": results[0].names[cls],
                "confidence": conf
            })
    
    # 显示处理完成信息
    st.success(f"批量处理完成！结果保存在: {results_dir}")
    
    # 显示汇总统计
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 显示类别分布
        st.subheader("检测类别分布")
        fig = px.pie(df, names="class", title="所有图片的检测类别分布")
        st.plotly_chart(fig)
        
        # 显示详细结果表格
        st.subheader("详细检测结果")
        st.dataframe(df)
        
        # 添加下载按钮
        csv_path = os.path.join(results_dir, "detection_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        with open(csv_path, "rb") as f:
            st.download_button(
                label="下载汇总结果",
                data=f,
                file_name="detection_results.csv",
                mime="text/csv"
            )

def infer_uploaded_image(conf, model):
    """处理上传的图片
    Args:
        conf: 置信度阈值
        model: YOLO模型实例
    """
    # 创建文件上传器，支持多种图片格式
    source_img = st.file_uploader(
        "上传图片", type=("jpg", "jpeg", "png", "bmp", "webp")
    )
    
    # 添加批量处理选项
    st.subheader("批量处理")
    folder_path = st.text_input("输入图片文件夹路径")
    if folder_path and os.path.isdir(folder_path):
        if st.button("开始批量处理"):
            process_batch_images(folder_path, conf, model)
    
    if source_img is not None:
        # 打开上传的图片
        uploaded_image = Image.open(source_img)
        
        # 创建两列布局，用于显示原图和检测结果
        col1, col2 = st.columns(2)
        
        with col1:
            # 显示原图
            st.image(source_img, caption="上传的图片", use_container_width=True)
        
        # 添加检测按钮
        if st.button("开始检测"):
            with st.spinner("正在处理..."):
                # 使用模型进行预测
                results = model.predict(uploaded_image, conf=conf)
                # 处理预测结果，将BGR转换为RGB
                result_image = results[0].plot()[:, :, ::-1]
                # 在第二列显示检测结果
                with col2:
                    st.image(result_image, caption="检测结果", use_container_width=True)
                
                # 保存检测结果
                json_path = save_detection_results(results, source_img.name)
                st.success(f"检测结果已保存到: {json_path}")
                
                # 在图片下方显示统计信息（占据整个页面宽度）
                st.markdown("---")  # 添加分隔线
                st.subheader("检测统计信息")
                visualize_detection_results(results, conf)
                
                # 添加下载按钮
                with open(json_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="下载检测结果",
                        data=f,
                        file_name=f"{source_img.name.split('.')[0]}_results.json",
                        mime="application/json"
                    )

def infer_uploaded_video(conf, model):
    """处理上传的视频
    Args:
        conf: 置信度阈值
        model: YOLO模型实例
    """
    # 创建视频文件上传器
    source_video = st.file_uploader(
        "上传视频", type=("mp4", "avi", "mov")
    )
    
    if source_video is not None:
        # 创建临时文件保存上传的视频
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(source_video.read())
        
        # 添加检测按钮
        if st.button("开始检测"):
            with st.spinner("正在处理..."):
                try:
                    # 打开视频文件
                    vid_cap = cv2.VideoCapture(tfile.name)
                    # 创建视频显示区域
                    stframe = st.empty()
                    
                    # 创建进度条
                    progress_bar = st.progress(0)
                    frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # 创建结果保存目录
                    os.makedirs("results", exist_ok=True)
                    output_path = f"results/{source_video.name.split('.')[0]}_processed.mp4"
                    
                    # 获取视频信息
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
                    
                    # 创建视频写入器
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    # 逐帧处理视频
                    frame_idx = 0
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            # 对每一帧进行预测
                            results = model.predict(image, conf=conf)
                            # 处理预测结果
                            result_image = results[0].plot()
                            
                            # 显示处理后的帧
                            stframe.image(result_image, caption="检测结果", channels="BGR", use_container_width=True)
                            
                            # 写入处理后的帧
                            out.write(result_image)
                            
                            # 更新进度条
                            frame_idx += 1
                            progress_bar.progress(frame_idx / frame_count)
                        else:
                            vid_cap.release()
                            out.release()
                            break
                    
                    # 显示处理完成信息
                    st.success(f"视频处理完成，已保存到: {output_path}")
                    
                    # 添加下载按钮
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="下载处理后的视频",
                            data=f,
                            file_name=f"{source_video.name.split('.')[0]}_processed.mp4",
                            mime="video/mp4"
                        )
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                finally:
                    # 删除临时文件
                    os.unlink(tfile.name)

def infer_uploaded_webcam(conf, model):
    """处理摄像头输入
    Args:
        conf: 置信度阈值
        model: YOLO模型实例
    """
    # 创建控制按钮
    col1, col2 = st.columns(2)
    start_button = col1.button("开始检测")
    stop_button = col2.button("停止检测")
    
    if start_button:
        try:
            # 打开摄像头（0表示默认摄像头）
            vid_cap = cv2.VideoCapture(0)
            # 创建视频显示区域
            stframe = st.empty()
            
            # 创建结果保存目录
            os.makedirs("results", exist_ok=True)
            output_path = f"results/webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            st.success(f"视频将要保存到: results/webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            # 获取摄像头信息
            width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 实时处理摄像头输入
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    # 对每一帧进行预测
                    results = model.predict(image, conf=conf)
                    # 处理预测结果，将BGR转换为RGB
                    result_image = results[0].plot()[:, :, ::-1]
                    # 显示处理后的帧
                    stframe.image(result_image, caption="检测结果", use_container_width=True)
                    # 写入处理后的帧
                    out.write(cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                    
                    # 检查是否点击停止按钮
                    if stop_button:
                        break
                else:
                    break
                    
            # 释放资源
            vid_cap.release()
            out.release()
            
            # 显示处理完成信息
            st.success(f"视频已保存到: {output_path}")
            
            # 添加下载按钮
            with open(output_path, "rb") as f:
                st.download_button(
                    label="下载录制的视频",
                    data=f,
                    file_name=f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4"
                )                
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")
