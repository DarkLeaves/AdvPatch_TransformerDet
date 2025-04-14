import numpy as np
import torch
from .base import BaseAttacker
from torch.optim import Optimizer
import torch.nn.functional as F

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


class OptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    # @property
    # def param_groups(self):
    #     return self.optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def patch_update(self, **kwargs):
        self.optimizer.step()
        # grad = self.optimizer.param_groups[0]['params'][0].grad
        # print(torch.mean(torch.abs(grad)))
        self.patch_obj.clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)

    def attack_loss(self, confs):
        self.optimizer.zero_grad()
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0])
        tv_loss, obj_loss = loss.values()
        tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
        loss = tv_loss.to(obj_loss.device) + obj_loss
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out


class CFAOptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    # @property
    # def param_groups(self):
    #     return self.optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def patch_update(self, **kwargs):
        self.optimizer.step()
        # grad = self.optimizer.param_groups[0]['params'][0].grad
        # print(torch.mean(torch.abs(grad)))
        self.patch_obj.clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)

    # fi
    #     def non_targeted_attack(self, ori_tensor_batch, detector, **kwargs):
    #         losses = []
    #         for iter in range(self.iter_step):
    #             if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()

    #             # Aggregate Gradient
    #             ens = 10
    #             agg_grad = 0
    #             for e in range(ens):
    #                 ori_tensor_batch_copy = ori_tensor_batch.clone()
    #                 mask = np.random.binomial(1, 0.7, size=ori_tensor_batch_copy.size())
    #                 mask_tensor = torch.from_numpy(mask).to(detector.device)
    #                 ori_tensor_batch_copy = ori_tensor_batch_copy * mask_tensor
    #                 adv_tensor_batch_ag = self.detector_attacker.uap_apply(ori_tensor_batch_copy)
    #                 adv_tensor_batch_ag = adv_tensor_batch_ag.to(detector.device)
    #                 bboxes, confs, cls_array = detector(adv_tensor_batch_ag).values()
    #                 self.detector_attacker.clear_fm_grad()
    #                 detector.zero_grad()
    #                 # attack_cls = int(self.cfg.ATTACK_CLASS)
    #                 # conf = torch.cat(([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in
    #                 #                    zip(confs, cls_array)]))
    #                 # loss = conf.mean()
    #                 loss = confs.max(dim=-1, keepdim=True)[0].mean()
    #                 loss.backward()
    #                 agg_grad += self.detector_attacker.mid_back_grad

    #             # print('len of mid_back_grad: ', len(self.detector_attacker.mid_back_grad))
    #             # assert len(self.detector_attacker.mid_back_grad) == ori_tensor_batch.size()[0]
    #             # feature_weights = torch.stack(self.detector_attacker.mid_back_grad, dim=0).mean(dim=0)
    #             # feature_weights = agg_grad / ens
    #             squared = agg_grad ** 2
    #             square = squared.sum(dim=(1, 2, 3), keepdim=True)
    #             # square = square.clamp(square, min=1e-4)
    #             feature_weights = agg_grad / (torch.rsqrt(square) + 1e-7)
    #             self.detector_attacker.clear_fm_grad()

    #             adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
    #             adv_tensor_batch = adv_tensor_batch.to(detector.device)
    #             # detect adv img batch to get bbox and obj confs
    #             bboxes, confs, cls_array = detector(adv_tensor_batch).values()

    #             if hasattr(self.cfg, 'class_specify'):
    #                 # TODO: only support filtering a single class now
    #                 attack_cls = int(self.cfg.ATTACK_CLASS)
    #                 confs = torch.cat(
    #                     ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
    #             elif hasattr(self.cfg, 'topx_conf'):
    #                 # attack top x confidence
    #                 # print(confs.size())
    #                 confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
    #                 confs = torch.mean(confs, dim=-1)
    #             else:
    #                 # only attack the max confidence
    #                 confs = confs.max(dim=-1, keepdim=True)[0]

    #             detector.zero_grad()
    #             # print('confs', confs)
    #             loss_dict = self.attack_loss(confs=confs, feature_map=self.detector_attacker.mid_feature_map,
    #                                          weights=feature_weights)
    #             loss = loss_dict['loss']
    #             # print(loss)
    #             loss.backward()
    #             # print(self.detector_attacker.patch_obj.patch.grad)
    #             losses.append(float(loss))

    #             # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
    #             self.patch_update()
    #         # print(adv_tensor_batch, bboxes, loss_dict)
    #         # update training statistics on tensorboard
    #         self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
    #         return torch.tensor(losses).mean()

    #     def attack_loss(self, confs, **kwargs):
    #         feature_map = kwargs['feature_map']
    #         weights = kwargs['weights']
    #         self.optimizer.zero_grad()
    #         loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0], weights=weights,
    #                             feature_map=feature_map)
    #         tv_loss, obj_loss, fi_loss = loss.values()
    #         tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
    #         loss = tv_loss.to(obj_loss.device) + obj_loss + fi_loss
    #         out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss, 'fi_loss': fi_loss}
    #         return out

    def non_targeted_attack(self, ori_tensor_batch, detector, **kwargs):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()

            ori_tensor_batch2 = ori_tensor_batch.clone().requires_grad_(True)
            self.detector_attacker.clear_fm_grad()
            detector.zero_grad()
            ori_tensor_batch2 = ori_tensor_batch2.to(detector.device)
            bboxes, confs, cls_array = detector(ori_tensor_batch2).values()

            # for yolov2
            # attack_cls = 0
            # print(confs)
            # print(cls_array)
            # # confs = torch.cat(([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            # selected_confs = [conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]
            # selected_confs = torch.cat(selected_confs) if selected_confs else torch.tensor([0.0],requires_grad=True).to(confs.device)

            # for yolov5
            selected_confs = confs.max(dim=-1, keepdim=True)[0]
            temp_loss = selected_confs.mean()
            temp_loss.backward(retain_graph=True)

            mid_feature = self.detector_attacker.mid_feature_map
            mid_back_grad = self.detector_attacker.mid_back_grad
            deep_feature = self.detector_attacker.deep_feature_map
            deep_back_grad = self.detector_attacker.deep_back_grad

            weighted_mid_feature = mid_feature * mid_back_grad
            weighted_deep_feature = deep_feature * deep_back_grad

            resize_weighted_deep_feature = weighted_deep_feature.view(weighted_deep_feature.shape[0],
                                                                      weighted_mid_feature.shape[1],
                                                                      weighted_deep_feature.shape[1] // weighted_mid_feature.shape[1],
                                                                      weighted_deep_feature.shape[2],
                                                                      weighted_deep_feature.shape[3]).mean(dim=2)
            weighted_deep_feature = F.interpolate(resize_weighted_deep_feature, size=(weighted_mid_feature.shape[2],
                                                                                      weighted_mid_feature.shape[3]),
                                                                                      mode='bilinear',
                                                                                      align_corners=False)

            p_mask = (weighted_deep_feature > 0) & (weighted_mid_feature > 0)
            p_feature = weighted_mid_feature * weighted_deep_feature * p_mask.float()
            # n_mask = (weighted_deep_feature < 0) & (weighted_mid_feature < 0)
            # n_feature = weighted_mid_feature * weighted_deep_feature * n_mask.float()
            # feature_loss = p_feature.mean() + n_feature.mean()
            feature_loss = p_feature.mean()

            # p_values = torch.zeros_like(weighted_feature)
            # n_values = torch.zeros_like(weighted_feature)
            # p_values[weighted_feature > 0] = weighted_feature[weighted_feature > 0]
            # n_values[weighted_feature < 0] = weighted_feature[weighted_feature < 0]
            # feature_loss = p_values.mean() + n_values.mean()
            # feature_loss = p_values.mean()
            # feature_loss = n_values.mean()

            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                # TODO: only support filtering a single class now
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                # attack top x confidence
                # print(confs.size())
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                # only attack the max confidence
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            loss_dict = self.attack_loss(confs=confs, extr_loss=feature_loss)
            loss = loss_dict['loss']
            loss.backward()
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    def attack_loss(self, confs, **kwargs):
        extr_loss = kwargs['extr_loss']
        self.optimizer.zero_grad()
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0], extr_loss=extr_loss)
        tv_loss, obj_loss, extr_loss = loss.values()
        tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
        loss = tv_loss.to(obj_loss.device) + obj_loss + extr_loss
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss, 'extr_loss': extr_loss}
        # tv_loss, obj_loss, fi_loss = loss.values()
        # tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
        # loss = tv_loss.to(obj_loss.device) + obj_loss + fi_loss
        # out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss, 'fi_loss': fi_loss}
        return out



class TransOptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    # @property
    # def param_groups(self):
    #     return self.optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def patch_update(self, **kwargs):
        self.optimizer.step()
        # grad = self.optimizer.param_groups[0]['params'][0].grad
        # print(torch.mean(torch.abs(grad)))
        self.patch_obj.clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)

    def attack_loss(self, confs):
        self.optimizer.zero_grad()
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0])
        tv_loss, obj_loss = loss.values()
        tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
        loss = tv_loss.to(obj_loss.device) + obj_loss
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()


            # only attack the max confidence
            # print('max confidence')
            confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            # print('confs', confs)
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            # print(loss)
            loss.backward()
            # print('patch_obj.patch.grad:')
            # print(self.detector_attacker.patch_obj.patch.grad)
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()