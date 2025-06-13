# 图像处理系统

这是一个基于 Streamlit 开发的综合性图像处理系统，集成了多种图像处理功能，包括目标检测、图像处理、人脸处理等。

## 功能特点

- 图像处理
  - 图像融合
  - 图像增强
  - 图像滤波
  - 图像分割
  - OCR文字识别

- 目标检测（YOLOv8）
  - 支持检测/分割/关键点三种任务
  - 支持图片/视频/摄像头输入
  - 可配置置信度阈值
  - 支持多种预训练模型

- 人脸处理
  - 人脸检测
  - 唇色调整
  - 实时视频处理

## 环境要求

- Python 3.8+
- CUDA 11.8+ (用于GPU加速，可选)
- Tesseract-OCR (用于OCR功能)

## 环境配置

1. 创建并激活虚拟环境：
```bash
conda create -n image python=3.8
conda activate image
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

3. 安装 Tesseract-OCR：
- Windows: 从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装
- 安装时请确保选中"Additional language data (download)"以支持中文识别
- 将 Tesseract 安装路径添加到系统环境变量 PATH 中

## 项目结构

```
├── Page/                    # 页面模块
│   ├── image_process_page.py  # 图像处理页面
│   ├── yolo_page.py          # YOLO检测页面
│   ├── face_lip_adjust.py    # 人脸唇色调整
│   └── ...
├── utils.py                 # 工具函数
├── config.py               # 配置文件
├── requirements.txt        # 依赖包列表
└── README.txt             # 项目说明
```

## 使用方法

1. 启动应用：
```bash
streamlit run index.py
```

2. 在浏览器中访问：
```
http://localhost:8501
```

3. 使用说明：
   - 图像处理：选择处理类型，上传图片，调整参数，点击处理按钮
   - 目标检测：选择模型和任务类型，上传图片或视频，或使用摄像头
   - 人脸处理：上传图片或使用摄像头，调整参数

## 注意事项

1. 首次运行时需要下载预训练模型，请确保网络连接正常
2. 使用摄像头功能时，请确保设备已正确连接
3. OCR功能需要正确安装 Tesseract-OCR 和中文语言包
4. 建议使用 GPU 进行目标检测，可以显著提升处理速度

## 常见问题

1. 如果遇到 Tesseract 相关错误：
   - 检查 Tesseract 是否正确安装
   - 确认中文语言包是否已下载
   - 验证环境变量是否正确设置

2. 如果遇到 CUDA 相关错误：
   - 检查 CUDA 版本是否兼容
   - 确认 PyTorch 版本是否支持当前 CUDA 版本

3. 如果遇到摄像头访问错误：
   - 检查摄像头是否被其他程序占用
   - 确认摄像头驱动是否正确安装

## 依赖包列表

主要依赖包：
- streamlit
- opencv-python
- pytorch
- ultralytics
- face-recognition
- pytesseract
- numpy
- pillow
- plotly
- pandas

详细依赖请参考 requirements.txt

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。 