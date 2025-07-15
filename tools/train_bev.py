#!/usr/bin/env python3
"""
BEV 전용 VAD 모델 학습 스크립트
"""

import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import train_model
from mmcv.utils import get_git_hash


def parse_args():
    parser = argparse.ArgumentParser(description='BEV VAD 모델 학습')
    parser.add_argument('config', help='학습 설정 파일 경로')
    parser.add_argument('--work-dir', help='작업 디렉토리 경로')
    parser.add_argument('--load-from', help='체크포인트 로드 경로')
    parser.add_argument('--resume-from', help='재개할 체크포인트 경로')
    parser.add_argument('--no-validate', action='store_true',
                        help='학습 중 검증 비활성화')
    parser.add_argument('--gpus', type=int, default=1,
                        help='사용할 GPU 수')
    parser.add_argument('--gpu-ids', type=int, nargs='+',
                        help='사용할 GPU ID들')
    parser.add_argument('--seed', type=int, default=0,
                        help='랜덤 시드')
    parser.add_argument('--deterministic', action='store_true',
                        help='deterministic 모드 활성화')
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='설정 옵션 오버라이드')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='작업 런처')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true',
                        help='배치 크기에 따른 학습률 자동 조정')
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def main():
    args = parse_args()
    
    # 설정 파일 로드
    cfg = Config.fromfile(args.config)
    
    # 옵션 적용
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    # 플러그인 설정
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # 기본 플러그인 디렉토리
                import projects.mmdet3d_plugin
    
    # 작업 디렉토리 설정
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                               osp.splitext(osp.basename(args.config))[0])
    
    # 체크포인트 경로 설정
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    # GPU 설정
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    
    # 학습률 자동 조정
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8
    
    # 분산 학습 초기화
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    
    # 작업 디렉토리 생성
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # 로거 설정
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    # 환경 정보 로깅
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    
    # 설정 정보 로깅
    meta = dict()
    meta['env_info'] = env_info_dict
    meta['config'] = cfg.pretty_text
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)
    
    # 설정 파일 저장
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
    # 랜덤 시드 설정
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                   f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    
    # 모델 빌드
    model = build_model(cfg.model)
    model.init_weights()
    
    # 데이터셋 빌드
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    
    # 메타 정보 업데이트
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE if hasattr(datasets[0], 'PALETTE') else None)
    
    # 검증 설정
    if not args.no_validate:
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hooks = [dict(type='EvalHook', **eval_cfg)]
    else:
        eval_hooks = []
    
    # 학습 시작
    logger.info(f'Start training')
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main() 