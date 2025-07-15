#!/usr/bin/env python3
"""
BEV 전용 VAD 모델을 위한 더미 데이터셋 생성 스크립트
"""

import os
import pickle
import json
import numpy as np
from PIL import Image
import cv2
import mmcv
from tqdm import tqdm
import uuid
import random
from datetime import datetime


def generate_dummy_bev_image(size=(200, 200), save_path=None):
    """더미 BEV 이미지 생성"""
    # 랜덤 색상으로 도로 배경 생성
    bev_img = np.random.randint(50, 100, (*size, 3), dtype=np.uint8)
    
    # 도로 차선 추가
    center_x = size[0] // 2
    for i in range(0, size[1], 20):
        cv2.line(bev_img, (center_x, i), (center_x, i+10), (255, 255, 255), 2)
    
    # 랜덤 차량 박스 추가
    num_vehicles = random.randint(3, 8)
    for _ in range(num_vehicles):
        x = random.randint(10, size[0]-30)
        y = random.randint(10, size[1]-30)
        w = random.randint(15, 25)
        h = random.randint(35, 45)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        cv2.rectangle(bev_img, (x, y), (x+w, y+h), color, -1)
    
    # 저장
    if save_path:
        cv2.imwrite(save_path, bev_img)
    
    return bev_img


def generate_dummy_3d_boxes(num_boxes=5, pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]):
    """더미 3D 박스 생성"""
    boxes = []
    labels = []
    velocities = []
    
    class_names = ['car', 'truck', 'bus', 'trailer', 'pedestrian', 'bicycle', 'motorcycle']
    
    for i in range(num_boxes):
        # 랜덤 위치 (BEV 좌표계)
        x = random.uniform(pc_range[0], pc_range[3])
        y = random.uniform(pc_range[1], pc_range[4])
        z = random.uniform(pc_range[2], pc_range[5])
        
        # 랜덤 크기
        w = random.uniform(1.5, 2.5)
        l = random.uniform(3.5, 5.0)
        h = random.uniform(1.5, 2.0)
        
        # 랜덤 회전
        yaw = random.uniform(-np.pi, np.pi)
        
        # 랜덤 속도
        vx = random.uniform(-5.0, 5.0)
        vy = random.uniform(-5.0, 5.0)
        
        boxes.append([x, y, z, w, l, h, yaw, vx, vy])
        labels.append(random.randint(0, len(class_names)-1))
        velocities.append([vx, vy])
    
    return np.array(boxes), np.array(labels), np.array(velocities)


def generate_dummy_map_data(num_lines=10):
    """더미 맵 데이터 생성"""
    map_lines = []
    map_labels = []
    
    map_classes = ['divider', 'ped_crossing', 'boundary']
    
    for i in range(num_lines):
        # 랜덤 라인 생성
        num_points = 20  # 고정된 포인트 수
        
        # 시작점과 끝점 설정
        start_x = random.uniform(-15, 15)
        start_y = random.uniform(-30, 30)
        end_x = random.uniform(-15, 15)
        end_y = random.uniform(-30, 30)
        
        # 선형 보간으로 포인트 생성
        x_points = np.linspace(start_x, end_x, num_points)
        y_points = np.linspace(start_y, end_y, num_points)
        
        # 노이즈 추가
        x_points += np.random.normal(0, 0.5, num_points)
        y_points += np.random.normal(0, 0.5, num_points)
        
        # 포인트 배열 생성
        points = []
        for x, y in zip(x_points, y_points):
            points.extend([x, y])
        
        map_lines.append(points)
        map_labels.append(random.randint(0, len(map_classes)-1))
    
    return map_lines, np.array(map_labels)


def generate_dummy_sample_info(token, timestamp, data_root):
    """더미 샘플 정보 생성"""
    return {
        'token': token,
        'timestamp': timestamp,
        'scene_token': f'scene_{token[:8]}',
        'sample_idx': random.randint(0, 1000),
        'ego2global_translation': [
            random.uniform(-1000, 1000),
            random.uniform(-1000, 1000),
            random.uniform(-5, 5)
        ],
        'ego2global_rotation': [
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ],
        'lidar2ego_translation': [0.0, 0.0, 1.84],
        'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
        'map_location': 'boston-seaport',
        'frame_idx': random.randint(0, 100),
        'bev_image_path': f'bev_images/{token}.png',
        'gt_names': ['car', 'truck', 'bus', 'trailer', 'pedestrian'],
        'gt_boxes': [],
        'gt_labels': [],
        'gt_velocities': [],
        'map_gt_lines': [],
        'map_gt_labels': []
    }


def create_dummy_dataset(data_root, num_samples=100):
    """더미 데이터셋 생성"""
    print(f"Creating dummy dataset at {data_root}")
    
    # 디렉토리 생성
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(os.path.join(data_root, 'bev_images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'bev_images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'bev_images', 'test'), exist_ok=True)
    
    # 데이터 분할
    train_samples = int(num_samples * 0.7)
    val_samples = int(num_samples * 0.2)
    test_samples = num_samples - train_samples - val_samples
    
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    all_infos = {'train': [], 'val': [], 'test': []}
    
    for split, num_split_samples in splits.items():
        print(f"Generating {split} split: {num_split_samples} samples")
        
        for i in tqdm(range(num_split_samples)):
            # 토큰 생성
            token = str(uuid.uuid4().hex)
            timestamp = int(datetime.now().timestamp() * 1000000) + i
            
            # BEV 이미지 생성
            bev_img_path = os.path.join(data_root, 'bev_images', split, f'{token}.png')
            generate_dummy_bev_image(save_path=bev_img_path)
            
            # 3D 박스 생성
            gt_boxes, gt_labels, gt_velocities = generate_dummy_3d_boxes()
            
            # 맵 데이터 생성
            map_lines, map_labels = generate_dummy_map_data()
            
            # 샘플 정보 생성
            sample_info = generate_dummy_sample_info(token, timestamp, data_root)
            sample_info['gt_boxes'] = gt_boxes.tolist()
            sample_info['gt_labels'] = gt_labels.tolist()
            sample_info['gt_velocities'] = gt_velocities.tolist()
            sample_info['map_gt_lines'] = map_lines
            sample_info['map_gt_labels'] = map_labels.tolist()
            
            all_infos[split].append(sample_info)
    
    # 정보 파일 저장
    for split, infos in all_infos.items():
        info_file = os.path.join(data_root, f'vad_nuscenes_infos_{split}.pkl')
        with open(info_file, 'wb') as f:
            pickle.dump(infos, f)
        print(f"Saved {len(infos)} {split} samples to {info_file}")
    
    # 맵 annotation 파일 생성
    map_ann_data = {
        'GTs': {},
        'meta': {
            'map_classes': ['divider', 'ped_crossing', 'boundary'],
            'pc_range': [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            'fixed_ptsnum_per_line': 20
        }
    }
    
    # 각 샘플에 대한 맵 GT 생성
    for split in ['train', 'val', 'test']:
        for info in all_infos[split]:
            token = info['token']
            map_ann_data['GTs'][token] = {
                'gt_vecs_label': info['map_gt_labels'],
                'gt_vecs_pts_loc': info['map_gt_lines'],
                'gt_vecs_pts_num': [20] * len(info['map_gt_lines'])
            }
    
    map_ann_file = os.path.join(data_root, 'nuscenes_map_anns_val.json')
    with open(map_ann_file, 'w') as f:
        json.dump(map_ann_data, f, indent=2)
    print(f"Saved map annotations to {map_ann_file}")
    
    print("Dummy dataset creation completed!")
    return all_infos


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='더미 BEV 데이터셋 생성')
    parser.add_argument('--data-root', default='data/nuscenes_dummy', 
                       help='데이터 저장 경로')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='생성할 샘플 수')
    
    args = parser.parse_args()
    
    # 더미 데이터셋 생성
    all_infos = create_dummy_dataset(args.data_root, args.num_samples)
    
    # 생성된 데이터 정보 출력
    print("\n=== 생성된 데이터 요약 ===")
    for split, infos in all_infos.items():
        print(f"{split}: {len(infos)} samples")
        if infos:
            sample = infos[0]
            print(f"  - Sample keys: {list(sample.keys())}")
            print(f"  - GT boxes shape: {len(sample['gt_boxes'])}")
            print(f"  - Map lines: {len(sample['map_gt_lines'])}")
    
    print(f"\n데이터 위치: {args.data_root}")
    print("다음 명령어로 학습 테스트 가능:")
    print(f"python tools/test_bev_pipeline.py --data-root {args.data_root}")


if __name__ == '__main__':
    main() 