如何训练红外目标检测数据集，红外深度学习目标检测中使用YOLOv8架构进行红外行人和车辆目标检测任务，训练使用识别红外行人车辆数据集
@[toc]


以下文字及代码仅供参考。

红外行人和车辆数据集，红外目标检测数据集
假如你有_如何训练？
8000张照片，训练集6400张，验证集1600张，txt格式

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/73f29c224d254198a01f703a25b57eae.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/68d1956c6b13401fa180f16ddb21abef.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a2ab3871bdf342639e36d8d3134abd92.png)
使用YOLOv8架构进行红外行人和车辆目标检测任务，如何准备数据、构建项目结构、编写训练脚本、评估脚本、推理脚本及可视化脚本。以下是具体步骤：

### 数据集目录结构

首先，确保你的数据集按照以下结构组织：

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

其中`data.yaml`文件定义了数据集的路径和类别信息：

```yaml
train: ./dataset/images/train/
val: ./dataset/images/val/

nc: 2  # 类别数量
names: ['person', 'vehicle']  # 类别名
```

### 安装依赖项

确保安装了YOLOv8及其相关依赖项：

```bash
pip install ultralytics
```

### 数据准备

每个图像都有一个对应的`.txt`文件，该文件包含边界框信息。每行代表一个对象，格式如下：

```
class_id center_x center_y width height
```

所有值都是相对于图像尺寸归一化后的浮点数。

### 训练脚本

创建一个Python脚本来开始训练过程：

```python
from ultralytics import YOLO

def main_train():
    # 加载YOLOv8模型，'yolov8n.yaml'表示使用nano版本，你可以选择其他尺寸如's', 'm', 'l', 'x'
    model = YOLO('yolov8n.yaml')  # 或者直接加载预训练权重，例如'ultralytics/yolov8n.pt'
    
    results = model.train(
        data='./path/to/data.yaml',  # 替换为你的data.yaml路径
        epochs=100,  # 根据需要调整训练周期数
        imgsz=640,  # 图像尺寸
        batch=16,  # 批大小，根据你的硬件条件调整
        project='./runs/detect',
        name='infrared_object_detection',
        optimizer='SGD',
        device='0',  # 使用GPU编号，'0'表示第一个GPU
        save=True,
        cache=True,
    )

if __name__ == '__main__':
    main_train()
```

### 评估脚本

训练完成后，可以使用验证集对模型进行评估：

```python
from ultralytics import YOLO

model = YOLO('./runs/detect/infrared_object_detection/weights/best.pt')
metrics = model.val(data='./path/to/data.yaml')
print(metrics.box.map)  # 输出mAP值等指标
```

### 推理与可视化脚本

加载训练好的模型进行推理，并可视化结果：

```python
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO('./runs/detect/infrared_object_detection/weights/best.pt')

def detect_objects(image_path):
    results = model.predict(source=image_path)
    img = cv2.imread(image_path)
    for result in results:
        boxes = result.boxes.numpy()
        for box in boxes:
            r = box.xyxy
            x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            label = result.names[int(box.cls)]
            confidence = box.conf
            if confidence > 0.5:  # 设置置信度阈值
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制矩形框
                cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

# 示例调用
result_image = detect_objects('your_test_image.jpg')
Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).show()  # 使用PIL显示图像
```

### 运行步骤总结

#### 步骤 1: 准备数据集

- 确保数据集按照上述目录结构组织。
- 创建并配置`data.yaml`文件。

#### 步骤 2: 安装依赖

```bash
pip install ultralytics
```

#### 步骤 3: 训练模型

运行训练脚本：

```bash
python train.py
```

#### 步骤 4: 模型评估

在验证集上评估模型性能：

```bash
python evaluate.py
```

#### 步骤 5: 推理与可视化

对新图像进行预测并可视化结果：

```bash
python inference_visualize.py --image_path your_test_image.jpg
```


