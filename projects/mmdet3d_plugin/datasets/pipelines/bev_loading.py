import os
import numpy as np
import cv2
from PIL import Image
import torch
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadImageFromFile, Resize, Normalize, Pad
from mmcv.parallel import DataContainer as DC


@PIPELINES.register_module()
class LoadBEVImage:
    """BEV 이미지 로딩"""
    
    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type
    
    def __call__(self, results):
        """BEV 이미지 로딩 실행"""
        if 'bev_image' in results:
            # 이미 로드된 BEV 이미지가 있으면 사용
            bev_image = results['bev_image']
            if self.to_float32:
                bev_image = bev_image.astype(np.float32)
        else:
            # 파일에서 BEV 이미지 로드
            bev_path = results.get('bev_path', None)
            if bev_path and os.path.exists(bev_path):
                bev_image = cv2.imread(bev_path)
                if self.color_type == 'color':
                    bev_image = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)
                if self.to_float32:
                    bev_image = bev_image.astype(np.float32)
            else:
                # 기본값으로 빈 이미지 생성
                bev_image = np.zeros((200, 200, 3), dtype=np.float32)
        
        results['bev_image'] = bev_image
        results['bev_shape'] = bev_image.shape
        results['bev_fields'] = ['bev_image']
        
        return results


@PIPELINES.register_module()
class BEVResize:
    """BEV 이미지 리사이즈"""
    
    def __init__(self, img_scale, keep_ratio=True):
        self.img_scale = img_scale
        self.keep_ratio = keep_ratio
    
    def __call__(self, results):
        """BEV 이미지 리사이즈 실행"""
        bev_image = results['bev_image']
        
        if self.keep_ratio:
            # 비율 유지하며 리사이즈
            h, w = bev_image.shape[:2]
            target_w, target_h = self.img_scale
            
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized_image = cv2.resize(bev_image, (new_w, new_h))
            
            # 패딩으로 목표 크기 맞추기
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            
            padded_image = np.pad(resized_image, 
                                ((pad_h, target_h - new_h - pad_h),
                                 (pad_w, target_w - new_w - pad_w),
                                 (0, 0)), 
                                mode='constant', constant_values=0)
            
            results['bev_image'] = padded_image
        else:
            # 직접 리사이즈
            resized_image = cv2.resize(bev_image, self.img_scale)
            results['bev_image'] = resized_image
        
        results['bev_shape'] = results['bev_image'].shape
        results['scale_factor'] = getattr(results, 'scale_factor', 1.0)
        
        return results


@PIPELINES.register_module()
class BEVNormalize:
    """BEV 이미지 정규화"""
    
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
    
    def __call__(self, results):
        """BEV 이미지 정규화 실행"""
        bev_image = results['bev_image']
        
        if self.to_rgb:
            bev_image = bev_image[..., ::-1]  # BGR to RGB
        
        # 정규화
        bev_image = (bev_image - self.mean) / self.std
        
        results['bev_image'] = bev_image
        results['bev_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        
        return results


@PIPELINES.register_module()
class BEVPad:
    """BEV 이미지 패딩"""
    
    def __init__(self, size_divisor=32, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val
    
    def __call__(self, results):
        """BEV 이미지 패딩 실행"""
        bev_image = results['bev_image']
        
        h, w = bev_image.shape[:2]
        
        # 목표 크기 계산
        target_h = ((h + self.size_divisor - 1) // self.size_divisor) * self.size_divisor
        target_w = ((w + self.size_divisor - 1) // self.size_divisor) * self.size_divisor
        
        # 패딩 크기 계산
        pad_h = target_h - h
        pad_w = target_w - w
        
        if pad_h > 0 or pad_w > 0:
            padded_image = np.pad(bev_image, 
                                ((0, pad_h), (0, pad_w), (0, 0)),
                                mode='constant', constant_values=self.pad_val)
            results['bev_image'] = padded_image
        
        results['bev_shape'] = results['bev_image'].shape
        results['pad_shape'] = results['bev_image'].shape
        
        return results


@PIPELINES.register_module()
class FormatBEVBundle3D:
    """BEV 데이터 포맷팅"""
    
    def __init__(self, class_names, collect_keys=None):
        self.class_names = class_names
        self.collect_keys = collect_keys or []
    
    def __call__(self, results):
        """BEV 데이터 포맷팅 실행"""
        # BEV 이미지를 텐서로 변환
        bev_image = results['bev_image']
        if len(bev_image.shape) == 3:
            bev_image = bev_image.transpose(2, 0, 1)  # HWC to CHW
        
        bev_tensor = torch.from_numpy(bev_image).float()
        
        # 결과 딕셔너리 생성
        formatted_results = {}
        
        # BEV 이미지 추가
        if 'bev_image' in self.collect_keys:
            formatted_results['bev_image'] = DC(bev_tensor, stack=True)
        
        # 메타데이터 추가
        img_metas = {}
        for key in ['bev_shape', 'pad_shape', 'scale_factor', 'bev_norm_cfg']:
            if key in results:
                img_metas[key] = results[key]
        
        # 기본 메타데이터 복사
        for key in ['sample_idx', 'token', 'timestamp', 'ego2global_translation',
                   'ego2global_rotation', 'lidar2ego_translation', 'lidar2ego_rotation']:
            if key in results:
                img_metas[key] = results[key]
        
        formatted_results['img_metas'] = DC(img_metas, cpu_only=True)
        
        # GT 데이터 추가
        if 'gt_bboxes_3d' in self.collect_keys and 'gt_bboxes_3d' in results:
            formatted_results['gt_bboxes_3d'] = DC(results['gt_bboxes_3d'], cpu_only=True)
        
        if 'gt_labels_3d' in self.collect_keys and 'gt_labels_3d' in results:
            formatted_results['gt_labels_3d'] = DC(
                torch.from_numpy(results['gt_labels_3d']).long(), cpu_only=False)
        
        # 맵 GT 데이터 추가
        if 'map_gt_bboxes_3d' in self.collect_keys and 'map_gt_bboxes_3d' in results:
            formatted_results['map_gt_bboxes_3d'] = DC(results['map_gt_bboxes_3d'], cpu_only=True)
        
        if 'map_gt_labels_3d' in self.collect_keys and 'map_gt_labels_3d' in results:
            formatted_results['map_gt_labels_3d'] = DC(
                torch.from_numpy(results['map_gt_labels_3d']).long(), cpu_only=False)
        
        # 속성 레이블 추가
        if 'gt_attr_labels' in self.collect_keys and 'gt_attr_labels' in results:
            formatted_results['gt_attr_labels'] = DC(
                torch.from_numpy(results['gt_attr_labels']).long(), cpu_only=False)
        
        return formatted_results


@PIPELINES.register_module()
class BEVRandomFlip:
    """BEV 이미지 랜덤 플립"""
    
    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
    
    def __call__(self, results):
        """BEV 이미지 랜덤 플립 실행"""
        if np.random.rand() < self.flip_ratio:
            bev_image = results['bev_image']
            
            if self.direction == 'horizontal':
                bev_image = np.flip(bev_image, axis=1)
            elif self.direction == 'vertical':
                bev_image = np.flip(bev_image, axis=0)
            
            results['bev_image'] = bev_image.copy()
            results['flip'] = True
            results['flip_direction'] = self.direction
        else:
            results['flip'] = False
        
        return results


@PIPELINES.register_module()
class BEVRandomRotate:
    """BEV 이미지 랜덤 회전"""
    
    def __init__(self, rotate_ratio=0.5, angle_range=(-10, 10)):
        self.rotate_ratio = rotate_ratio
        self.angle_range = angle_range
    
    def __call__(self, results):
        """BEV 이미지 랜덤 회전 실행"""
        if np.random.rand() < self.rotate_ratio:
            bev_image = results['bev_image']
            h, w = bev_image.shape[:2]
            
            # 랜덤 각도 생성
            angle = np.random.uniform(*self.angle_range)
            
            # 회전 행렬 생성
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 이미지 회전
            rotated_image = cv2.warpAffine(bev_image, rotation_matrix, (w, h))
            
            results['bev_image'] = rotated_image
            results['rotate'] = True
            results['rotate_angle'] = angle
        else:
            results['rotate'] = False
        
        return results 