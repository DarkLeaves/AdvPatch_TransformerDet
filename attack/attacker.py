import torch
import numpy as np
import os
import multiprocessing
from utils.det_utils import plot_boxes_cv2
from utils import FormatConverter
from detlib.utils import init_detectors
from scripts.dict import get_attack_method, loss_dict
from utils import DataTransformer, pad_lab
from attack.uap import PatchManager, PatchRandomApplier
from utils.det_utils import inter_nms

# for debug
# def save_tensor_img(save_path, img_tensor):
#     import torchvision.transforms as transforms
#     from PIL import Image
#     from datetime import datetime
#     import os
#
#     os.makedirs(save_path, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#     if img_tensor.max() <= 1:
#         img_tensor = img_tensor * 255
#
#     img_tensor = img_tensor.byte()  # 转换为 8-bit 格式
#
#     for i, tensor in enumerate(img_tensor):
#         img = transforms.ToPILImage()(tensor)
#         img_path = os.path.join(save_path, f"idx_{i}_{timestamp}.png")
#         img.save(img_path)
#         print('Save Success at: ', img_path)


class UniversalAttacker(object):
    """An attacker agent to coordinate the detect & base attack methods for universal attacks."""

    def __init__(self, cfg, device: torch.device):
        """

        :param cfg: Parsed proj config object.
        :param device: torch.device, cpu or cuda
        """
        self.cfg = cfg
        self.device = device
        self.max_boxes = 15
        self.patch_boxes = []

        self.class_names = cfg.all_class_names  # class names reference: labels of all the classes
        self.attack_list = cfg.attack_list  # int list: classes index to be attacked, [40, 41, 42, ...]
        self.patch_obj = PatchManager(cfg.ATTACKER.PATCH, device)
        self.vlogger = None

        self.patch_applier = PatchRandomApplier(device, cfg_patch=cfg.ATTACKER.PATCH)
        self.detectors = init_detectors(cfg_det=cfg.DETECTOR)

        # differentiable Kornia augmentation method
        # self.data_transformer = DataTransformer(device, rand_rotate=0)

    @property
    def universal_patch(self):
        """ This is for convenient calls.

        :return: the adversarial patch tensor.
        """
        return self.patch_obj.patch

    def init_attaker(self):
        cfg = self.cfg.ATTACKER
        loss_fn = loss_dict[cfg.LOSS_FUNC]
        self.attacker = get_attack_method(cfg.METHOD)(
            loss_func=loss_fn, norm='L_infty', device=self.device, cfg=cfg, detector_attacker=self)

    def plot_boxes(self, img_tensor, boxes, save_path=None, save_name=None):
        """Plot detected boxes on images.

        :param img_tensor: a singe image tensor.
        :param boxes: bounding boxes of the img_tensor.
        :param save_path: save path.
        :param save_name: save name of the plotted image.
        :return: plotted image.
        """
        # print(img.dtype, isinstance(img, np.ndarray))
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, save_name)
        img = FormatConverter.tensor2numpy_cv2(img_tensor.cpu().detach())
        plot_box = plot_boxes_cv2(img, boxes.cpu().detach().numpy(), self.class_names,
                                  savename=save_name)
        return plot_box

    def init_universal_patch(self, patch_file=None):
        self.patch_obj.init(patch_file)
        # self.universal_patch = self.patch_obj.patch

    def filter_bbox(self, preds, target_cls=None):
        """To filter classes.
            FIXME: To be a more universal op fn. Support only a single target class currently.
        :param preds:
        :param target_cls:
        :return:
        """

        if len(preds) == 0: return preds
        # if cls_array is None: cls_array = preds[:, -1]
        # filt = [cls in self.cfg.attack_list for cls in cls_array]
        # preds = preds[filt]
        target_cls = self.cfg.attack_cls if target_cls is None else target_cls
        return preds[preds[:, -1] == target_cls]

    def get_patch_pos_batch(self, all_preds):
        """To filter bboxes of the given target class. If none target bbox is got, return has_target=False

        :param all_preds: all predection results
        :return: number of target boxes
        """

        self.all_preds = all_preds
        batch_boxes = None
        target_nums = []
        for i_batch, preds in enumerate(all_preds):
            if len(preds) == 0:
                preds = torch.cuda.FloatTensor([[0, 0, 0, 0, 0, 0]])
            preds = self.filter_bbox(preds)
            padded_boxs = pad_lab(preds, self.max_boxes).unsqueeze(0)
            batch_boxes = padded_boxs if batch_boxes is None else torch.vstack((batch_boxes, padded_boxs))
            target_nums.append(len(preds))
        self.all_preds = batch_boxes
        return np.array(target_nums)

    def uap_apply(self, img_tensor: torch.Tensor, adv_patch: torch.Tensor = None):
        """To attach the uap(universal adversarial patch) onto the image samples.
        :param img_tensor: image batch tensor.
        :param adv_patch: adversarial patch tensor.
        :return:
        """
        if adv_patch is None: adv_patch = self.universal_patch
        img_tensor = self.patch_applier(img_tensor, adv_patch, self.all_preds)


        # 1st inference: get bbox; 2rd inference: get detections of the adversarial patch

        # If you wanna augment data here, use the provided differentiable Kornia augmentation method:
        # img_tensor = self.data_transformer(img_tensor)

        return img_tensor

    def merge_batch(self, all_preds, preds):
        """To merge detection results.

        :param all_preds:
        :param preds:
        :return:
        """
        if all_preds is None:
            return preds
        for i, (all_pred, pred) in enumerate(zip(all_preds, preds)):
            if pred.shape[0]:
                pred = pred.to(all_pred.device)
                all_preds[i] = torch.cat((all_pred, pred), dim=0)
                continue
            all_preds[i] = all_pred if all_pred.shape[0] else pred
        return all_preds

    def detect_bbox(self, img_batch, detectors=None):
        if detectors is None:
            detectors = self.detectors

        all_preds = None
        for detector in detectors:
            preds = detector(img_batch.to(detector.device))['bbox_array']
            all_preds = self.merge_batch(all_preds, preds)

        # nms among detectors
        if len(detectors) > 1: all_preds = inter_nms(all_preds)
        return all_preds

    def attack(self, img_tensor_batch, mode='sequential'):
        """Call the base attack method to optimize the patch.

        :param img_tensor_batch: image batch input.
        :param mode: attack mode(To define the updating behavior of multi-model ensembling.)
        :return: loss
        """
        # import matplotlib.pyplot as plt
        # t = img_tensor_batch.cpu()
        # for idx in range(len(t)):
        #     temo = t[idx].permute(1, 2, 0).numpy()
        #     name = 'mg_read_' + str(idx) + '.png'
        #     plt.imsave(name,temo)
        detectors_loss = []
        self.attacker.begin_attack()
        if mode == 'optim' or mode == 'sequential':
            for detector in self.detectors:
                loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
                detectors_loss.append(loss)
        elif mode == 'parallel':
            detectors_loss = self.parallel_attack(img_tensor_batch)
        self.attacker.end_attack()
        return torch.tensor(detectors_loss).mean()
        # lmy:  why not:
        # return torch.tensor(detectors_loss).sum()

    def parallel_attack(self, img_tensor_batch: torch.Tensor):
        """Multi-model ensembling: parallel attack mode.
            To average multi-updates to obtain the ultimate patch update in a single iter.
            FIXME: Not fully-supported currently.
            (ps. Not used in T-SEA.)

        :param img_tensor_batch:
        :return: loss
        """
        detectors_loss = []
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            patch_tmp, loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
            patch_update = patch_tmp - self.universal_patch
            patch_updates += patch_update
            detectors_loss.append(loss)
        self.patch_obj.update_((self.universal_patch + patch_updates / len(self.detectors)).detach_())
        return detectors_loss


class CFAUniversalAttacker(object):
    """An attacker to attack CNN-based detectors with adversarial patch."""

    def __init__(self, cfg, device: torch.device):
        """

        :param cfg: Parsed proj config object.
        :param device: torch.device, cpu or cuda
        """
        self.cfg = cfg
        self.device = device
        self.max_boxes = 15
        self.patch_boxes = []

        self.class_names = cfg.all_class_names  # class names reference: labels of all the classes
        self.attack_list = cfg.attack_list  # int list: classes index to be attacked, [40, 41, 42, ...]
        self.patch_obj = PatchManager(cfg.ATTACKER.PATCH, device)
        self.vlogger = None

        self.patch_applier = PatchRandomApplier(device, cfg_patch=cfg.ATTACKER.PATCH)
        self.detectors = init_detectors(cfg_det=cfg.DETECTOR)

        self.mid_feature_map = 0
        self.mid_back_grad = 0
        self.deep_feature_map = 0
        self.deep_back_grad = 0

        # differentiable Kornia augmentation method
        # self.data_transformer = DataTransformer(device, rand_rotate=0)

    def regist_hook(self):

        def mid_back_hook(module, grad_in, grad_out):
            self.mid_back_grad = (list(grad_in)[0])

        def mid_forward_hook(module, input, output):
            self.mid_feature_map = output
        def deep_back_hook(module, grad_in, grad_out):
            self.deep_back_grad = (list(grad_in)[0])

        def deep_forward_hook(module, input, output):
            self.deep_feature_map = output

        assert len(self.detectors) == 1, "only work for attacking one single detector now."
        detector = self.detectors[0]
        if detector.name == 'yolov2':
            detector.detector.models[6].register_forward_hook(mid_forward_hook)
            detector.detector.models[7].register_full_backward_hook(mid_back_hook)
            print('Forward and Backward hooks registered.')
        elif detector.name == 'yolov3':
            pass
        elif detector.name == 'yolov3-tiny':
            pass
        elif detector.name == 'yolov4':
            pass
        elif detector.name == 'yolov4-tiny':
            pass
        elif detector.name == 'yolov5':
            # deep feature from conv block 5
            detector.detector.model[5].register_forward_hook(deep_forward_hook)
            detector.detector.model[6].register_full_backward_hook(deep_back_hook)

            # deep feature from block 6
            # detector.detector.model[6].register_forward_hook(deep_forward_hook)
            # detector.detector.model[7].register_full_backward_hook(deep_back_hook)

            # deep feature from block 7
            # detector.detector.model[7].register_forward_hook(deep_forward_hook)
            # detector.detector.model[8].register_full_backward_hook(deep_back_hook)

            # deep feature from block 8
            # detector.detector.model[8].register_forward_hook(deep_forward_hook)
            # detector.detector.model[9].register_full_backward_hook(deep_back_hook)

            # mid feature from block 3
            detector.detector.model[3].register_forward_hook(mid_forward_hook)
            detector.detector.model[4].register_full_backward_hook(mid_back_hook)
            print('Forward and Backward hooks registered.')
        elif detector.name == 'faster_rcnn':
            pass
        elif detector.name == 'ssd':
            pass
        else:
            pass

    # @property
    def clear_fm_grad(self):
        self.mid_feature_map = 0
        self.mid_back_grad = 0
        self.deep_feature_map = 0
        self.deep_back_grad = 0
        return

    @property
    def universal_patch(self):
        """ This is for convenient calls.

        :return: the adversarial patch tensor.
        """
        return self.patch_obj.patch

    def init_attaker(self):
        cfg = self.cfg.ATTACKER
        loss_fn = loss_dict[cfg.LOSS_FUNC]
        self.attacker = get_attack_method(cfg.METHOD)(
            loss_func=loss_fn, norm='L_infty', device=self.device, cfg=cfg, detector_attacker=self)

    def plot_boxes(self, img_tensor, boxes, save_path=None, save_name=None):
        """Plot detected boxes on images.

        :param img_tensor: a singe image tensor.
        :param boxes: bounding boxes of the img_tensor.
        :param save_path: save path.
        :param save_name: save name of the plotted image.
        :return: plotted image.
        """
        # print(img.dtype, isinstance(img, np.ndarray))
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, save_name)
        img = FormatConverter.tensor2numpy_cv2(img_tensor.cpu().detach())
        plot_box = plot_boxes_cv2(img, boxes.cpu().detach().numpy(), self.class_names,
                                  savename=save_name)
        return plot_box

    def init_universal_patch(self, patch_file=None):
        self.patch_obj.init(patch_file)
        # self.universal_patch = self.patch_obj.patch

    def filter_bbox(self, preds, target_cls=None):
        """To filter classes.
            FIXME: To be a more universal op fn. Support only a single target class currently.
        :param preds:
        :param target_cls:
        :return:
        """

        if len(preds) == 0: return preds
        # if cls_array is None: cls_array = preds[:, -1]
        # filt = [cls in self.cfg.attack_list for cls in cls_array]
        # preds = preds[filt]
        target_cls = self.cfg.attack_cls if target_cls is None else target_cls
        return preds[preds[:, -1] == target_cls]

    def get_patch_pos_batch(self, all_preds):
        """To filter bboxes of the given target class. If none target bbox is got, return has_target=False

        :param all_preds: all predection results
        :return: number of target boxes
        """

        self.all_preds = all_preds
        batch_boxes = None
        target_nums = []
        for i_batch, preds in enumerate(all_preds):
            if len(preds) == 0:
                preds = torch.cuda.FloatTensor([[0, 0, 0, 0, 0, 0]])
            preds = self.filter_bbox(preds)
            padded_boxs = pad_lab(preds, self.max_boxes).unsqueeze(0)
            batch_boxes = padded_boxs if batch_boxes is None else torch.vstack((batch_boxes, padded_boxs))
            target_nums.append(len(preds))
        self.all_preds = batch_boxes
        return np.array(target_nums)

    def uap_apply(self, img_tensor: torch.Tensor, adv_patch: torch.Tensor = None):
        """To attach the uap(universal adversarial patch) onto the image samples.
        :param img_tensor: image batch tensor.
        :param adv_patch: adversarial patch tensor.
        :return:
        """
        if adv_patch is None: adv_patch = self.universal_patch
        img_tensor = self.patch_applier(img_tensor, adv_patch, self.all_preds)

        # 1st inference: get bbox; 2rd inference: get detections of the adversarial patch

        # If you wanna augment data here, use the provided differentiable Kornia augmentation method:
        # img_tensor = self.data_transformer(img_tensor)

        return img_tensor

    def merge_batch(self, all_preds, preds):
        """To merge detection results.

        :param all_preds:
        :param preds:
        :return:
        """
        if all_preds is None:
            return preds
        for i, (all_pred, pred) in enumerate(zip(all_preds, preds)):
            if pred.shape[0]:
                pred = pred.to(all_pred.device)
                all_preds[i] = torch.cat((all_pred, pred), dim=0)
                continue
            all_preds[i] = all_pred if all_pred.shape[0] else pred
        return all_preds

    def detect_bbox(self, img_batch, detectors=None):
        if detectors is None:
            detectors = self.detectors

        all_preds = None
        for detector in detectors:
            preds = detector(img_batch.to(detector.device))['bbox_array']
            all_preds = self.merge_batch(all_preds, preds)

        # nms among detectors
        if len(detectors) > 1: all_preds = inter_nms(all_preds)
        return all_preds

    def attack(self, img_tensor_batch, mode='sequential'):
        """Call the base attack method to optimize the patch.

        :param img_tensor_batch: image batch input.
        :param mode: attack mode(To define the updating behavior of multi-model ensembling.)
        :return: loss
        """
        detectors_loss = []
        self.attacker.begin_attack()
        if mode == 'optim' or mode == 'sequential':
            for detector in self.detectors:
                loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
                detectors_loss.append(loss)
        elif mode == 'parallel':
            detectors_loss = self.parallel_attack(img_tensor_batch)
        self.attacker.end_attack()
        return torch.tensor(detectors_loss).mean()

    def parallel_attack(self, img_tensor_batch: torch.Tensor):
        """Multi-model ensembling: parallel attack mode.
            To average multi-updates to obtain the ultimate patch update in a single iter.
            FIXME: Not fully-supported currently.
            (ps. Not used in T-SEA.)

        :param img_tensor_batch:
        :return: loss
        """
        detectors_loss = []
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            patch_tmp, loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
            patch_update = patch_tmp - self.universal_patch
            patch_updates += patch_update
            detectors_loss.append(loss)
        self.patch_obj.update_((self.universal_patch + patch_updates / len(self.detectors)).detach_())
        return detectors_loss



class TransUniversalAttacker(object):
    """An attacker agent to coordinate the detect & base attack methods for universal attacks."""

    def __init__(self, cfg, device: torch.device):
        """

        :param cfg: Parsed proj config object.
        :param device: torch.device, cpu or cuda
        """
        self.cfg = cfg
        self.device = device
        self.max_boxes = 15
        self.patch_boxes = []

        self.class_names = cfg.all_class_names  # class names reference: labels of all the classes
        self.attack_list = cfg.attack_list  # int list: classes index to be attacked, [40, 41, 42, ...]
        self.patch_obj = PatchManager(cfg.ATTACKER.PATCH, device)
        self.vlogger = None

        self.patch_applier = PatchRandomApplier(device, cfg_patch=cfg.ATTACKER.PATCH)
        self.detectors = init_detectors(cfg_det=cfg.DETECTOR)

        # differentiable Kornia augmentation method
        # self.data_transformer = DataTransformer(device, rand_rotate=0)

    @property
    def universal_patch(self):
        """ This is for convenient calls.

        :return: the adversarial patch tensor.
        """
        return self.patch_obj.patch

    def init_attaker(self):
        cfg = self.cfg.ATTACKER
        loss_fn = loss_dict[cfg.LOSS_FUNC]
        self.attacker = get_attack_method(cfg.METHOD)(
            loss_func=loss_fn, norm='L_infty', device=self.device, cfg=cfg, detector_attacker=self)

    def plot_boxes(self, img_tensor, boxes, save_path=None, save_name=None):
        """Plot detected boxes on images.

        :param img_tensor: a singe image tensor.
        :param boxes: bounding boxes of the img_tensor.
        :param save_path: save path.
        :param save_name: save name of the plotted image.
        :return: plotted image.
        """
        # print(img.dtype, isinstance(img, np.ndarray))
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, save_name)
        img = FormatConverter.tensor2numpy_cv2(img_tensor.cpu().detach())
        plot_box = plot_boxes_cv2(img, boxes.cpu().detach().numpy(), self.class_names,
                                  savename=save_name)
        return plot_box

    def init_universal_patch(self, patch_file=None):
        self.patch_obj.init(patch_file)
        # self.universal_patch = self.patch_obj.patch

    def filter_bbox(self, preds, target_cls=None):
        """To filter classes.
            FIXME: To be a more universal op fn. Support only a single target class currently.
        :param preds:
        :param target_cls:
        :return:
        """

        if len(preds) == 0: return preds
        # if cls_array is None: cls_array = preds[:, -1]
        # filt = [cls in self.cfg.attack_list for cls in cls_array]
        # preds = preds[filt]
        target_cls = self.cfg.attack_cls if target_cls is None else target_cls
        return preds[preds[:, -1] == target_cls]

    def get_patch_pos_batch(self, all_preds):
        """To filter bboxes of the given target class. If none target bbox is got, return has_target=False

        :param all_preds: all predection results
        :return: number of target boxes
        """

        self.all_preds = all_preds
        batch_boxes = None
        target_nums = []
        for i_batch, preds in enumerate(all_preds):
            if len(preds) == 0:
                preds = torch.cuda.FloatTensor([[0, 0, 0, 0, 0, 0]])
            preds = self.filter_bbox(preds)
            padded_boxs = pad_lab(preds, self.max_boxes).unsqueeze(0)
            batch_boxes = padded_boxs if batch_boxes is None else torch.vstack((batch_boxes, padded_boxs))
            target_nums.append(len(preds))
        self.all_preds = batch_boxes
        return np.array(target_nums)


    def uap_apply(self, img_tensor: torch.Tensor, adv_patch: torch.Tensor = None):
        """To attach the uap(universal adversarial patch) onto the image samples.
        :param img_tensor: image batch tensor.
        :param adv_patch: adversarial patch tensor.
        :return:
        """
        if adv_patch is None: adv_patch = self.universal_patch
        img_tensor = self.patch_applier(img_tensor, adv_patch, self.all_preds)

        # print('img_tensor shape: ', img_tensor.shape)
        # save_tensor_img(save_path='temp', img_tensor=img_tensor)

        # 1st inference: get bbox; 2rd inference: get detections of the adversarial patch

        # If you wanna augment data here, use the provided differentiable Kornia augmentation method:
        # img_tensor = self.data_transformer(img_tensor)

        return img_tensor

    def merge_batch(self, all_preds, preds):
        """To merge detection results.

        :param all_preds:
        :param preds:
        :return:
        """
        if all_preds is None:
            return preds
        for i, (all_pred, pred) in enumerate(zip(all_preds, preds)):
            if pred.shape[0]:
                pred = pred.to(all_pred.device)
                all_preds[i] = torch.cat((all_pred, pred), dim=0)
                continue
            all_preds[i] = all_pred if all_pred.shape[0] else pred
        return all_preds

    def detect_bbox(self, img_batch, detectors=None):
        if detectors is None:
            detectors = self.detectors

        all_preds = None
        for detector in detectors:
            preds = detector(img_batch.to(detector.device))['bbox_array']
            all_preds = self.merge_batch(all_preds, preds)

        # nms among detectors
        if len(detectors) > 1: all_preds = inter_nms(all_preds)
        return all_preds

    def attack(self, img_tensor_batch, mode='sequential'):
        """Call the base attack method to optimize the patch.

        :param img_tensor_batch: image batch input.
        :param mode: attack mode(To define the updating behavior of multi-model ensembling.)
        :return: loss
        """

        detectors_loss = []
        self.attacker.begin_attack()
        if mode == 'optim' or mode == 'sequential':
            for detector in self.detectors:
                loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
                detectors_loss.append(loss)
        elif mode == 'parallel':
            detectors_loss = self.parallel_attack(img_tensor_batch)
        self.attacker.end_attack()
        return torch.tensor(detectors_loss).mean()
        # lmy:  why not:
        # return torch.tensor(detectors_loss).sum()

    # def parallel_attack(self, img_tensor_batch: torch.Tensor):
    #     """Multi-model ensembling: parallel attack mode.
    #         To average multi-updates to obtain the ultimate patch update in a single iter.
    #         FIXME: Not fully-supported currently.
    #         (ps. Not used in T-SEA.)
    #
    #     :param img_tensor_batch:
    #     :return: loss
    #     """
    #     detectors_loss = []
    #     patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
    #     for detector in self.detectors:
    #         patch_tmp, loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
    #         patch_update = patch_tmp - self.universal_patch
    #         patch_updates += patch_update
    #         detectors_loss.append(loss)
    #     self.patch_obj.update_((self.universal_patch + patch_updates / len(self.detectors)).detach_())
    #     return detectors_loss

    def regist_hook(self):

        def mid_back_hook(module, grad_in, grad_out):
            self.mid_back_grad = (list(grad_in)[0])

        def mid_forward_hook(module, input, output):
            self.mid_feature_map = output
        def deep_back_hook(module, grad_in, grad_out):
            self.deep_back_grad = (list(grad_in)[0])

        def deep_forward_hook(module, input, output):
            self.deep_feature_map = output

        assert len(self.detectors) == 1, "only work for attacking one single detector now."
        detector = self.detectors[0]
        if detector.name == 'yolov2':
            detector.detector.models[6].register_forward_hook(mid_forward_hook)
            detector.detector.models[7].register_full_backward_hook(mid_back_hook)
            print('Forward and Backward hooks registered.')
        else:
            print('No hooks registered.')