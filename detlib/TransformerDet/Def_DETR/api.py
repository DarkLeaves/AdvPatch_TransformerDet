import torch
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from abc import ABC, abstractmethod
from ...base import DetectorBase

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


class DeformableDETRDetector(DetectorBase):
    def __init__(self, name, cfg, input_tensor_size: int, device: torch.device):
        super().__init__(name, cfg, input_tensor_size, device)
        self.processor = None

    def eval(self):
        self.detector.model.requires_grad(False)
        self.detector.model.eval()

    def zero_grad(self):
        self.detector.model.zero_grad()

    def load(self, model_weights: str = "SenseTime/deformable-detr", **args):
        self.detector = DeformableDetrForObjectDetection.from_pretrained(model_weights).to(self.device)
        # self.processor = AutoImageProcessor.from_pretrained(model_weights)
        self.processor = AutoImageProcessor.from_pretrained(model_weights)

    def __call__(self, batch_tensor: torch.Tensor, threshold=0.5, **kwargs):
        """
        Detection core function, get detection results by feedding the input image
        :param batch_tensor: image tensor [batch_size, channel, h, w]
        :return:
            box_array: list of bboxes(batch_size*N*6) [[x1, y1, x2, y2, conf, cls_id],..]
            !!! x1, y1, x2, y2 are between [0,1]
            detections_with_grad: confidence of the object
        """

        assert self.detector, "ERROR! Model not loaded. Call load() first."

        if isinstance(batch_tensor, Image.Image):  # 检查是否为 PIL 图像
            target_sizes = torch.tensor([batch_tensor.size[::-1]])

            batch_tensor = self.processor(images=batch_tensor, return_tensors="pt")
            batch_tensor = batch_tensor.to(self.device)

            outputs = self.detector(**batch_tensor)
            results = self.processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
            return results

        elif torch.is_tensor(batch_tensor):
            batch_tensor = batch_tensor.to(self.device)
            outputs = self.detector(batch_tensor)

            image_sizes = [tensor.shape[-2:] for tensor in batch_tensor]  # 获取每张图像的 (height, width)
            target_sizes = torch.tensor(image_sizes)  # 确保与 batch 维度匹配

            predictions = self.processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)


            # for batchsize prediction
            bs = len(predictions)
            max_objects = 15  # because UniversalAttacker.max_boxes = 15 in attacker.py
            result_api = torch.zeros((bs, max_objects, 6), device=predictions[0]['boxes'].device)
            score_api = torch.zeros((bs, max_objects, 1), device=predictions[0]['scores'].device)
            # label_api = torch.zeros((bs, max_objects, 1), device=predictions[0]['labels'].device)

            for i in range(bs):
                n = predictions[i]['boxes'].shape[0]  # numbers of detected objects
                if n == 0:  # no object, skip
                    continue

                boxes = predictions[i]['boxes']  # shape: [n, 4]
                scores = predictions[i]['scores'].unsqueeze(1)  # shape: [n, 1]
                labels = (predictions[i]['labels'] - 1).unsqueeze(1)  # shape: [n, 1]

                # ues target_sizes to normalize bbox
                height, width = target_sizes[i]
                norm_factors = torch.tensor([width, height, width, height], device=boxes.device)  # [4]
                boxes = boxes / norm_factors

                # to meet the requirements of class DetectorBase
                result = torch.cat([boxes, scores, labels], dim=1)  # shape: [n, 6]

                result_api[i, :n, :] = result
                score_api[i, :n, :] = scores
                # label_api[i, :n, :] = labels

            score_api = score_api.squeeze(-1)
            return {'bbox_array': result_api, 'obj_confs': score_api, "cls_max_ids": None}

        else:
            raise TypeError("Input must be a PIL Image or PyTorch tensor.")