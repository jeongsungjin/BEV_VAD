import os
import json
import copy
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import torch
from PIL import Image
import mmcv
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC
from .nuscenes_vad_dataset import VADCustomNuScenesDataset


@DATASETS.register_module()
class BEVNuScenesDataset(VADCustomNuScenesDataset):
    """BEV 이미지만을 사용하는 NuScenes 데이터셋"""
    
    def __init__(self, 
                 bev_image_dir=None,
                 bev_h=200,
                 bev_w=200,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bev_image_dir = bev_image_dir
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # BEV 이미지 전용 설정
        self.input_modality = dict(
            use_lidar=False,
            use_camera=False,  # 기존 멀티뷰 카메라 비활성화
            use_radar=False,
            use_map=False,
            use_external=False,
            use_bev_image=True  # BEV 이미지 활성화
        )
    
    def load_bev_image(self, info):
        """BEV 이미지 로드"""
        token = info['token']
        bev_path = os.path.join(self.bev_image_dir, f"{token}.png")
        
        if os.path.exists(bev_path):
            # BEV 이미지 로드
            bev_image = Image.open(bev_path).convert('RGB')
            bev_image = bev_image.resize((self.bev_w, self.bev_h))
            bev_array = np.array(bev_image).astype(np.float32)
            # HWC -> CHW 변환
            bev_array = bev_array.transpose(2, 0, 1)
            return bev_array
        else:
            # BEV 이미지가 없으면 빈 이미지 생성
            return np.zeros((3, self.bev_h, self.bev_w), dtype=np.float32)
    
    def get_data_info(self, index):
        """데이터 정보 가져오기 (BEV 이미지 포함)"""
        info = self.data_infos[index]
        
        # BEV 이미지 정보 추가
        info['bev_image'] = self.load_bev_image(info)
        
        # GT 데이터 로드
        if 'gt_boxes' in info:
            from mmdet3d.core import LiDARInstance3DBoxes
            gt_boxes_3d = LiDARInstance3DBoxes(
                np.array(info['gt_boxes']), 
                box_dim=9, 
                origin=(0.5, 0.5, 0.5)
            )
            info['gt_bboxes_3d'] = gt_boxes_3d
        
        if 'gt_labels' in info:
            info['gt_labels_3d'] = np.array(info['gt_labels'])
        
        if 'gt_velocities' in info:
            info['gt_velocities'] = np.array(info['gt_velocities'])
        
        # 속성 라벨 (더미 데이터용)
        if 'gt_labels' in info:
            info['attr_labels'] = np.zeros(len(info['gt_labels']), dtype=np.int64)
        
        return info
    
    def prepare_train_data(self, index):
        """학습 데이터 준비"""
        data_queue = []
        
        # 시간적 데이터 큐 생성
        prev_indexs_list = list(range(index-self.queue_length, index))
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        
        # 현재 프레임 처리
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
            
        self.pre_pipeline(input_dict)
        example = self.bev_pipeline(input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        
        if self.filter_empty_gt and \
                ((example is None or ~(example['gt_labels_3d']._data != -1).any()) or \
                 (example is None or ~(example['map_gt_labels_3d']._data != -1).any())):
            return None
            
        data_queue.insert(0, example)
        
        # 이전 프레임들 처리
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
                
            self.pre_pipeline(input_dict)
            example = self.bev_pipeline(input_dict)
            
            if example is None:
                return None
            data_queue.insert(0, copy.deepcopy(example))
            
        return self.union2one(data_queue)
    
    def bev_pipeline(self, input_dict):
        """BEV 이미지 처리 파이프라인"""
        # pipeline을 사용하여 전처리
        if hasattr(self, 'pipeline'):
            example = self.pipeline(input_dict)
        else:
            # 기본 처리
            example = {
                'bev_image': input_dict['bev_image'],
                'img_metas': input_dict,
                'gt_bboxes_3d': input_dict.get('gt_bboxes_3d', []),
                'gt_labels_3d': input_dict.get('gt_labels_3d', []),
            }
        
        return example
    
    def prepare_test_data(self, index):
        """테스트 데이터 준비"""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
            
        self.pre_pipeline(input_dict)
        example = self.bev_pipeline(input_dict)
        
        return example 