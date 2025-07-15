_base_ = [
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# BEV 이미지 설정
bev_h_ = 200
bev_w_ = 200
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

# 클래스 설정
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_classes = len(class_names)

# 맵 클래스 설정
map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

# 입력 모달리티 설정 (BEV 이미지만 사용)
input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
    use_bev_image=True  # BEV 이미지만 사용
)

# 모델 차원 설정
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
queue_length = 4
total_epochs = 60

# 데이터셋 설정
dataset_type = 'BEVNuScenesDataset'
data_root = 'data/nuscenes/'
ann_file_train = data_root + 'vad_nuscenes_infos_train.pkl'
ann_file_val = data_root + 'vad_nuscenes_infos_val.pkl'
ann_file_test = data_root + 'vad_nuscenes_infos_test.pkl'
map_ann_file = data_root + 'nuscenes_map_anns_val.json'

# BEV 이미지 정규화 설정
bev_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# 모델 설정
model = dict(
    type='VAD_BEV',
    use_grid_mask=True,
    video_test_mode=True,
    
    # BEV 이미지 처리용 백본
    bev_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    
    # BEV 이미지 처리용 넥
    bev_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True
    ),
    
    # VAD 헤드 설정
    pts_bbox_head=dict(
        type='VADHead',
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        tot_epoch=total_epochs,
        use_traj_lr_warmup=False,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_his_encoder=None,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        
        # 트랜스포머 설정들... (기존 VAD 헤드와 동일)
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=num_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        map_num_vec=map_num_vec,
        map_num_classes=map_num_classes,
        map_num_pts_per_vec=map_fixed_ptsnum_per_gt_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        pc_range=point_cloud_range,
        embed_dims=_dim_,
        num_query=900,
        num_reg_fcs=2,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_dn=True,
        with_ego_pos=True,
        match_with_velo=True,
        match_costs=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range),
        transformer=dict(
            type='VADTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='MyCustomBaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm',
                                     'cross_attn', 'norm',
                                     'ffn', 'norm'))
            ),
            decoder=dict(
                type='CustomTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MyCustomBaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm',
                                     'cross_attn', 'norm',
                                     'ffn', 'norm'))
            ),
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_,
            ),
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_plan=dict(type='PlanLoss', loss_weight=1.0),
    ),
    
    # 학습 설정
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range))),
    
    # 테스트 설정
    test_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            max_per_img=300,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_thresh=0.1,
            pc_range=point_cloud_range,
            use_rotate_nms=True,
            nms_thr=0.2))
)

# 데이터 파이프라인 설정
train_pipeline = [
    dict(type='LoadBEVImage', to_float32=True),
    dict(type='BEVResize', img_scale=(bev_w_, bev_h_), keep_ratio=True),
    dict(type='BEVNormalize', **bev_norm_cfg),
    dict(type='BEVPad', size_divisor=32),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='FormatBEVBundle3D', 
         class_names=class_names,
         collect_keys=['bev_image', 'gt_bboxes_3d', 'gt_labels_3d',
                      'map_gt_bboxes_3d', 'map_gt_labels_3d', 'gt_attr_labels']),
]

test_pipeline = [
    dict(type='LoadBEVImage', to_float32=True),
    dict(type='BEVResize', img_scale=(bev_w_, bev_h_), keep_ratio=True),
    dict(type='BEVNormalize', **bev_norm_cfg),
    dict(type='BEVPad', size_divisor=32),
    dict(type='FormatBEVBundle3D',
         class_names=class_names,
         collect_keys=['bev_image']),
]

# 데이터셋 설정
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        map_ann_file=map_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_image_dir=data_root + 'bev_images/train/',  # BEV 이미지 디렉토리
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        pc_range=point_cloud_range,
        map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
        map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
        padding_value=0,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        bev_image_dir=data_root + 'bev_images/val/',
        bev_size=(bev_h_, bev_w_),
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_test,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        bev_image_dir=data_root + 'bev_images/test/',
        bev_size=(bev_h_, bev_w_),
        box_type_3d='LiDAR'))

# 옵티마이저 설정
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'bev_backbone': dict(lr_mult=0.1),
            'bev_neck': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 학습 스케줄러 설정
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# 런타임 설정
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# 평가 설정
evaluation = dict(interval=1, pipeline=test_pipeline)

# 체크포인트 설정
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

# 로그 설정
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 기타 설정
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/VAD_bev_e2e'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(1)
seed = 0
deterministic = False 