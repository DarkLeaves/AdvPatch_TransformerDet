import torch
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor

# COCO 80 类别名称
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table",
    "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# 1. 加载 Deformable DETR 模型
model_name = "SenseTime/deformable-detr"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeformableDetrForObjectDetection.from_pretrained(model_name).to(device)
processor = DeformableDetrImageProcessor.from_pretrained(model_name)


# 2. 读取并预处理图片
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def preprocess_image(image):
    inputs = processor(images=image, return_tensors="pt")
    return inputs


# 3. 进行推理
def predict(image_path):
    image = load_image(image_path)
    inputs = preprocess_image(image)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return image, outputs


# 4. 解析预测结果
def visualize_predictions(image, outputs, threshold=0.45):
    scores = outputs.logits.softmax(-1)[0, :, :-1]  # 排除背景类别
    boxes = outputs.pred_boxes[0]  # 归一化的 bbox

    # 只保留分数高于阈值的目标
    keep = scores.max(-1).values > threshold
    scores = scores[keep]
    boxes = boxes[keep]
    labels = scores.argmax(-1)

    print(scores, boxes, labels)

    # 映射回原图坐标
    w, h = image.size
    boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)

    image = np.array(image)
    for box, label, score in zip(boxes, labels, scores.max(-1).values):
        center_x, center_y, width, height = box.cpu().numpy()
        x_min = int(center_x - width / 2)
        y_min = int(center_y - height / 2)
        x_max = int(center_x + width / 2)
        y_max = int(center_y + height / 2)

        label_idx = label.item() - 1  # 修正类别索引
        category = COCO_CLASSES[label_idx] if 0 <= label_idx < len(COCO_CLASSES) else f"Class {label.item()}"


        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        text = f"{category} ({score:.2f})"
        cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Detected: {category} with confidence {score:.2f}")

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# 5. 测试代码
image_path = '../../data/INRIAPerson/Train/pos/crop001230.png'
image, outputs = predict(image_path)
visualize_predictions(image, outputs)

