#!/usr/bin/env python3
"""
BEV 파이프라인 동작 테스트 스크립트
"""

import os
import sys
import argparse
import torch
import numpy as np
try:
    from mmcv import Config
except ImportError:
    from mmengine import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from mmcv.utils import get_git_hash
from mmdet3d.utils import get_root_logger
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_dataset_loading(cfg, logger):
    """데이터셋 로딩 테스트"""
    logger.info("=== 데이터셋 로딩 테스트 ===")
    
    try:
        # 데이터셋 빌드
        dataset = build_dataset(cfg.data.train)
        logger.info(f"✅ 데이터셋 빌드 성공: {len(dataset)} samples")
        
        # 첫 번째 샘플 로드
        sample = dataset[0]
        logger.info(f"✅ 샘플 로드 성공")
        
        # 샘플 구조 확인
        logger.info("샘플 구조:")
        for key, value in sample.items():
            if hasattr(value, 'data'):
                logger.info(f"  {key}: {type(value.data)} - {value.data.shape if hasattr(value.data, 'shape') else 'no shape'}")
            else:
                logger.info(f"  {key}: {type(value)}")
        
        # BEV 이미지 확인
        if 'bev_image' in sample:
            bev_data = sample['bev_image'].data
            logger.info(f"✅ BEV 이미지 확인: {bev_data.shape}, dtype: {bev_data.dtype}")
        else:
            logger.warning("⚠️ BEV 이미지를 찾을 수 없습니다")
        
        # GT 데이터 확인
        if 'gt_labels_3d' in sample:
            gt_labels = sample['gt_labels_3d'].data
            logger.info(f"✅ GT 라벨 확인: {gt_labels.shape}, dtype: {gt_labels.dtype}")
        
        if 'map_gt_labels_3d' in sample:
            map_labels = sample['map_gt_labels_3d'].data
            logger.info(f"✅ 맵 GT 라벨 확인: {map_labels.shape}, dtype: {map_labels.dtype}")
        
        return True, sample
        
    except Exception as e:
        logger.error(f"❌ 데이터셋 로딩 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_building(cfg, logger):
    """모델 빌드 테스트"""
    logger.info("=== 모델 빌드 테스트 ===")
    
    try:
        # 모델 빌드
        model = build_model(cfg.model)
        logger.info(f"✅ 모델 빌드 성공: {model.__class__.__name__}")
        
        # 모델 구조 확인
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ 모델 파라미터: {total_params:,} (trainable: {trainable_params:,})")
        
        # 모델 모드 설정
        model.train()
        
        return True, model
        
    except Exception as e:
        logger.error(f"❌ 모델 빌드 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model, sample, logger):
    """모델 순전파 테스트"""
    logger.info("=== 순전파 테스트 ===")
    
    try:
        # 배치 차원 추가
        batch_sample = {}
        for key, value in sample.items():
            if hasattr(value, 'data'):
                if key == 'bev_image':
                    # BEV 이미지는 (C, H, W) -> (1, 1, C, H, W) (batch, queue_length, channels, height, width)
                    batch_sample[key] = value.data.unsqueeze(0).unsqueeze(0)
                elif key == 'img_metas':
                    # 메타데이터는 리스트로 감싸기
                    batch_sample[key] = [[value.data]]
                else:
                    # 다른 데이터는 배치 차원만 추가
                    if hasattr(value.data, 'unsqueeze'):
                        batch_sample[key] = [value.data]
                    else:
                        batch_sample[key] = [value.data]
            else:
                batch_sample[key] = [value]
        
        logger.info("배치 데이터 준비 완료")
        
        # 순전파 실행 (학습 모드)
        with torch.no_grad():
            try:
                # 학습 모드 테스트
                model.train()
                logger.info("학습 모드 순전파 시작...")
                
                # 입력 데이터 형태 변환
                forward_inputs = {
                    'bev_img': batch_sample['bev_image'],
                    'img_metas': batch_sample['img_metas'],
                    'return_loss': True
                }
                
                # GT 데이터 추가
                if 'gt_bboxes_3d' in batch_sample:
                    forward_inputs['gt_bboxes_3d'] = batch_sample['gt_bboxes_3d']
                if 'gt_labels_3d' in batch_sample:
                    forward_inputs['gt_labels_3d'] = batch_sample['gt_labels_3d']
                if 'map_gt_bboxes_3d' in batch_sample:
                    forward_inputs['map_gt_bboxes_3d'] = batch_sample['map_gt_bboxes_3d']
                if 'map_gt_labels_3d' in batch_sample:
                    forward_inputs['map_gt_labels_3d'] = batch_sample['map_gt_labels_3d']
                
                # 순전파 실행
                outputs = model(**forward_inputs)
                
                if isinstance(outputs, dict):
                    logger.info(f"✅ 학습 모드 순전파 성공")
                    logger.info("출력 구조:")
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"  {key}: {value.shape}, dtype: {value.dtype}")
                        else:
                            logger.info(f"  {key}: {type(value)} - {value}")
                else:
                    logger.info(f"✅ 학습 모드 순전파 성공: {type(outputs)}")
                
            except Exception as e:
                logger.error(f"❌ 학습 모드 순전파 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
            
            try:
                # 추론 모드 테스트
                model.eval()
                logger.info("추론 모드 순전파 시작...")
                
                test_inputs = {
                    'bev_img': batch_sample['bev_image'],
                    'img_metas': batch_sample['img_metas'],
                    'return_loss': False
                }
                
                test_outputs = model(**test_inputs)
                logger.info(f"✅ 추론 모드 순전파 성공")
                
                if isinstance(test_outputs, (list, tuple)):
                    logger.info(f"추론 결과: {len(test_outputs)} 배치")
                    for i, result in enumerate(test_outputs):
                        if isinstance(result, dict):
                            logger.info(f"  배치 {i}: {list(result.keys())}")
                        else:
                            logger.info(f"  배치 {i}: {type(result)}")
                
            except Exception as e:
                logger.error(f"❌ 추론 모드 순전파 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 순전파 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(cfg, logger):
    """데이터로더 테스트"""
    logger.info("=== 데이터로더 테스트 ===")
    
    try:
        from mmdet3d.datasets import build_dataloader
        
        # 데이터셋 빌드
        dataset = build_dataset(cfg.data.train)
        
        # 데이터로더 빌드
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False,
            seed=0
        )
        
        logger.info(f"✅ 데이터로더 빌드 성공: {len(dataloader)} 배치")
        
        # 첫 번째 배치 로드
        batch_iter = iter(dataloader)
        batch = next(batch_iter)
        
        logger.info("배치 구조:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}, dtype: {value.dtype}")
            elif isinstance(value, list):
                logger.info(f"  {key}: list of {len(value)} items")
                if value and isinstance(value[0], torch.Tensor):
                    logger.info(f"    - item 0: {value[0].shape}, dtype: {value[0].dtype}")
            else:
                logger.info(f"  {key}: {type(value)}")
        
        return True, batch
        
    except Exception as e:
        logger.error(f"❌ 데이터로더 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='BEV 파이프라인 테스트')
    parser.add_argument('--config', 
                       default='projects/configs/VAD/VAD_bev_e2e.py',
                       help='설정 파일 경로')
    parser.add_argument('--data-root', 
                       default='data/nuscenes_dummy',
                       help='데이터 루트 경로')
    parser.add_argument('--device', default='cpu',
                       help='사용할 디바이스')
    
    args = parser.parse_args()
    
    # 로거 설정
    logger = get_root_logger(log_level='INFO')
    
    # 설정 파일 로드
    logger.info(f"설정 파일 로드: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # 데이터 경로 오버라이드
    cfg.data.train.data_root = args.data_root
    cfg.data.train.ann_file = os.path.join(args.data_root, 'vad_nuscenes_infos_train.pkl')
    cfg.data.train.map_ann_file = os.path.join(args.data_root, 'nuscenes_map_anns_val.json')
    cfg.data.train.bev_image_dir = os.path.join(args.data_root, 'bev_images/train/')
    
    # 플러그인 임포트
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            plg_lib = importlib.import_module(_module_path)
    
    # 디바이스 설정
    device = torch.device(args.device)
    
    logger.info(f"🚀 BEV 파이프라인 테스트 시작")
    logger.info(f"데이터 경로: {args.data_root}")
    logger.info(f"디바이스: {device}")
    
    # 테스트 실행
    test_results = []
    
    # 1. 데이터셋 로딩 테스트
    dataset_success, sample = test_dataset_loading(cfg, logger)
    test_results.append(("데이터셋 로딩", dataset_success))
    
    if not dataset_success:
        logger.error("데이터셋 로딩 실패로 테스트 중단")
        return
    
    # 2. 모델 빌드 테스트
    model_success, model = test_model_building(cfg, logger)
    test_results.append(("모델 빌드", model_success))
    
    if not model_success:
        logger.error("모델 빌드 실패로 테스트 중단")
        return
    
    # 3. 순전파 테스트
    if sample is not None and model is not None:
        model = model.to(device)
        forward_success = test_forward_pass(model, sample, logger)
        test_results.append(("순전파", forward_success))
    
    # 4. 데이터로더 테스트
    dataloader_success, batch = test_dataloader(cfg, logger)
    test_results.append(("데이터로더", dataloader_success))
    
    # 결과 요약
    logger.info("\n" + "="*50)
    logger.info("📊 테스트 결과 요약")
    logger.info("="*50)
    
    for test_name, success in test_results:
        status = "✅ 성공" if success else "❌ 실패"
        logger.info(f"{test_name}: {status}")
    
    all_success = all(success for _, success in test_results)
    
    if all_success:
        logger.info("\n🎉 모든 테스트 통과! 파이프라인이 정상적으로 작동합니다.")
        logger.info("다음 단계: 실제 학습 시작")
        logger.info(f"python tools/train_bev.py {args.config} --work-dir ./work_dirs/VAD_bev_test")
    else:
        logger.error("\n❌ 일부 테스트 실패. 코드를 확인해주세요.")
    
    logger.info("\n테스트 완료!")


if __name__ == '__main__':
    main() 