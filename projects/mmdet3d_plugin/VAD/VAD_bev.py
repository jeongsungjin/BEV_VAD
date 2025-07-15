import time
import copy
import torch
import torch.nn as nn
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric


@DETECTORS.register_module()
class VAD_BEV(MVXTwoStageDetector):
    """BEV 이미지만을 입력으로 받는 VAD 모델"""
    
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6,
                 bev_backbone=None,
                 bev_neck=None):
        
        super(VAD_BEV, self).__init__(
            pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained)
        
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.video_test_mode = video_test_mode
        
        # BEV 이미지 처리용 백본과 넥 추가
        self.bev_backbone = bev_backbone
        self.bev_neck = bev_neck
        
        if self.bev_backbone:
            self.bev_backbone = self._build_module(bev_backbone)
        if self.bev_neck:
            self.bev_neck = self._build_module(bev_neck)
    
    def _build_module(self, cfg):
        """모듈 빌드 헬퍼 함수"""
        from mmcv.cnn import build_model_from_cfg
        from mmdet.models import build_backbone, build_neck
        
        if cfg['type'] in ['ResNet', 'VoVNet']:
            return build_backbone(cfg)
        elif cfg['type'] in ['FPN']:
            return build_neck(cfg)
        else:
            return build_model_from_cfg(cfg)
    
    @auto_fp16(apply_to=('bev_img',), out_fp32=True)
    def extract_bev_feat(self, bev_img, img_metas=None):
        """BEV 이미지에서 특징 추출"""
        B, C, H, W = bev_img.shape
        
        # Grid mask 적용 (선택사항)
        if self.use_grid_mask:
            bev_img = self.grid_mask(bev_img)
        
        # BEV 백본을 통한 특징 추출
        if self.bev_backbone:
            bev_feats = self.bev_backbone(bev_img)
            if isinstance(bev_feats, (list, tuple)):
                bev_feats = bev_feats
            else:
                bev_feats = [bev_feats]
        else:
            # 백본이 없으면 원본 사용
            bev_feats = [bev_img]
        
        # BEV 넥 적용
        if self.bev_neck:
            bev_feats = self.bev_neck(bev_feats)
        
        return bev_feats
    
    @auto_fp16(apply_to=('bev_img',), out_fp32=True)
    def extract_feat(self, bev_img, img_metas=None, len_queue=None):
        """특징 추출 (BEV 이미지용)"""
        bev_feats = self.extract_bev_feat(bev_img, img_metas)
        return bev_feats
    
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          prev_bev=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None):
        """학습 시 포인트 포워드 함수"""
        
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels
        ]
        
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses
    
    def obtain_history_bev(self, bev_imgs_queue, img_metas_list):
        """BEV 히스토리 특징 획득"""
        self.eval()
        
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, C, H, W = bev_imgs_queue.shape
            bev_imgs_queue = bev_imgs_queue.reshape(bs * len_queue, C, H, W)
            
            bev_feats_list = self.extract_feat(bev_img=bev_imgs_queue, len_queue=len_queue)
            
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                bev_feats = [each_scale[i:i+1] for each_scale in bev_feats_list]
                prev_bev = self.pts_bbox_head(
                    bev_feats, img_metas, prev_bev, only_bev=True)
        
        self.train()
        return prev_bev
    
    @force_fp32(apply_to=('bev_img', 'points', 'prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      bev_img=None,  # BEV 이미지 입력
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      prev_bev=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None,
                      **kwargs):
        """학습 시 포워드 함수"""
        
        len_queue = bev_img.size(1)
        prev_bev = self.obtain_history_bev(bev_img, img_metas)
        
        # 현재 프레임 처리
        bev_img = bev_img[:, -1, ...]  # 마지막 프레임만 사용
        img_metas = [each[-1] for each in img_metas]
        
        # BEV 특징 추출
        bev_feats = self.extract_feat(bev_img=bev_img, img_metas=img_metas)
        
        # 손실 계산
        losses = self.forward_pts_train(
            bev_feats, gt_bboxes_3d, gt_labels_3d,
            map_gt_bboxes_3d, map_gt_labels_3d,
            img_metas, gt_bboxes_ignore, map_gt_bboxes_ignore,
            prev_bev, ego_his_trajs, ego_fut_trajs, ego_fut_masks,
            ego_fut_cmd, ego_lcf_feat, gt_attr_labels)
        
        return losses
    
    def forward_test(self,
                     bev_img=None,
                     img_metas=None,
                     **kwargs):
        """테스트 시 포워드 함수"""
        
        # 단일 이미지 테스트
        if bev_img.dim() == 4:  # [B, C, H, W]
            return self.simple_test(bev_img=bev_img, img_metas=img_metas, **kwargs)
        else:
            # 비디오 테스트 모드
            return self.forward_test_video(bev_img=bev_img, img_metas=img_metas, **kwargs)
    
    def simple_test(self, bev_img, img_metas, prev_bev=None, **kwargs):
        """단일 BEV 이미지 테스트"""
        
        # BEV 특징 추출
        bev_feats = self.extract_feat(bev_img=bev_img, img_metas=img_metas)
        
        # 예측 수행
        outs = self.pts_bbox_head(bev_feats, img_metas, prev_bev)
        
        # 결과 후처리
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=False)
        
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        
        return bbox_results
    
    def forward_test_video(self, bev_img, img_metas, **kwargs):
        """비디오 모드 테스트"""
        
        prev_bev = self.obtain_history_bev(bev_img, img_metas)
        
        # 현재 프레임 처리
        bev_img = bev_img[:, -1, ...]
        img_metas = [each[-1] for each in img_metas]
        
        return self.simple_test(bev_img=bev_img, img_metas=img_metas, 
                               prev_bev=prev_bev, **kwargs) 