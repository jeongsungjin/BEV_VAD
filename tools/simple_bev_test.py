#!/usr/bin/env python3
"""
간단한 BEV 데이터셋 동작 테스트 스크립트
"""

import os
import pickle
import json
import numpy as np
from PIL import Image
import cv2
import sys


def test_dummy_data_loading(data_root):
    """더미 데이터 로딩 테스트"""
    print("=== 더미 데이터 로딩 테스트 ===")
    
    # 1. 데이터 파일 존재 확인
    train_file = os.path.join(data_root, 'vad_nuscenes_infos_train.pkl')
    val_file = os.path.join(data_root, 'vad_nuscenes_infos_val.pkl')
    map_file = os.path.join(data_root, 'nuscenes_map_anns_val.json')
    
    if not os.path.exists(train_file):
        print(f"❌ 학습 데이터 파일이 없습니다: {train_file}")
        return False
    
    if not os.path.exists(val_file):
        print(f"❌ 검증 데이터 파일이 없습니다: {val_file}")
        return False
    
    if not os.path.exists(map_file):
        print(f"❌ 맵 데이터 파일이 없습니다: {map_file}")
        return False
    
    print("✅ 모든 필요한 파일이 존재합니다")
    
    # 2. 데이터 로드
    try:
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)
        print(f"✅ 학습 데이터 로드 성공: {len(train_data)} samples")
        
        with open(val_file, 'rb') as f:
            val_data = pickle.load(f)
        print(f"✅ 검증 데이터 로드 성공: {len(val_data)} samples")
        
        with open(map_file, 'r') as f:
            map_data = json.load(f)
        print(f"✅ 맵 데이터 로드 성공: {len(map_data['GTs'])} GT entries")
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {str(e)}")
        return False
    
    # 3. 데이터 구조 확인
    if train_data:
        sample = train_data[0]
        print(f"✅ 샘플 구조 확인:")
        print(f"  - 키: {list(sample.keys())}")
        print(f"  - 토큰: {sample['token']}")
        print(f"  - GT 박스 수: {len(sample['gt_boxes'])}")
        print(f"  - GT 라벨 수: {len(sample['gt_labels'])}")
        print(f"  - 맵 라인 수: {len(sample['map_gt_lines'])}")
        
        # BEV 이미지 확인
        bev_image_path = os.path.join(data_root, 'bev_images', 'train', f"{sample['token']}.png")
        if os.path.exists(bev_image_path):
            print(f"✅ BEV 이미지 확인: {bev_image_path}")
            
            # 이미지 로드 테스트
            try:
                img = cv2.imread(bev_image_path)
                if img is not None:
                    print(f"✅ BEV 이미지 로드 성공: {img.shape}")
                else:
                    print("❌ BEV 이미지 로드 실패")
                    return False
            except Exception as e:
                print(f"❌ BEV 이미지 로드 오류: {str(e)}")
                return False
        else:
            print(f"❌ BEV 이미지 파일이 없습니다: {bev_image_path}")
            return False
    
    return True


def test_bev_image_processing(data_root):
    """BEV 이미지 처리 테스트"""
    print("\n=== BEV 이미지 처리 테스트 ===")
    
    # 학습 데이터 로드
    train_file = os.path.join(data_root, 'vad_nuscenes_infos_train.pkl')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    if not train_data:
        print("❌ 학습 데이터가 없습니다")
        return False
    
    sample = train_data[0]
    bev_image_path = os.path.join(data_root, 'bev_images', 'train', f"{sample['token']}.png")
    
    try:
        # 이미지 로드
        img = cv2.imread(bev_image_path)
        print(f"✅ 원본 이미지 로드: {img.shape}")
        
        # RGB 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"✅ RGB 변환 성공: {img_rgb.shape}")
        
        # 크기 조정
        target_size = (200, 200)
        img_resized = cv2.resize(img_rgb, target_size)
        print(f"✅ 크기 조정 성공: {img_resized.shape}")
        
        # 정규화
        img_normalized = img_resized.astype(np.float32) / 255.0
        print(f"✅ 정규화 성공: {img_normalized.shape}, dtype: {img_normalized.dtype}")
        
        # 차원 변환 (HWC -> CHW)
        img_chw = img_normalized.transpose(2, 0, 1)
        print(f"✅ 차원 변환 성공: {img_chw.shape}")
        
        # 통계 확인
        print(f"✅ 픽셀 값 범위: [{img_chw.min():.3f}, {img_chw.max():.3f}]")
        print(f"✅ 평균값: {img_chw.mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 이미지 처리 실패: {str(e)}")
        return False


def test_gt_data_processing(data_root):
    """GT 데이터 처리 테스트"""
    print("\n=== GT 데이터 처리 테스트 ===")
    
    # 학습 데이터 로드
    train_file = os.path.join(data_root, 'vad_nuscenes_infos_train.pkl')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    if not train_data:
        print("❌ 학습 데이터가 없습니다")
        return False
    
    sample = train_data[0]
    
    try:
        # GT 박스 처리
        gt_boxes = np.array(sample['gt_boxes'])
        print(f"✅ GT 박스 처리: {gt_boxes.shape}")
        print(f"  - 박스 형태: {gt_boxes.dtype}")
        print(f"  - 박스 개수: {len(gt_boxes)}")
        
        # GT 라벨 처리
        gt_labels = np.array(sample['gt_labels'])
        print(f"✅ GT 라벨 처리: {gt_labels.shape}")
        print(f"  - 라벨 개수: {len(gt_labels)}")
        print(f"  - 라벨 범위: [{gt_labels.min()}, {gt_labels.max()}]")
        
        # 맵 데이터 처리
        map_lines = sample['map_gt_lines']
        map_labels = np.array(sample['map_gt_labels'])
        print(f"✅ 맵 데이터 처리:")
        print(f"  - 맵 라인 수: {len(map_lines)}")
        print(f"  - 맵 라벨 수: {len(map_labels)}")
        
        # 첫 번째 라인 확인
        if map_lines:
            first_line = map_lines[0]
            print(f"  - 첫 번째 라인 포인트 수: {len(first_line)//2}")
            print(f"  - 첫 번째 라인 데이터 타입: {type(first_line[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ GT 데이터 처리 실패: {str(e)}")
        return False


def test_data_consistency(data_root):
    """데이터 일관성 테스트"""
    print("\n=== 데이터 일관성 테스트 ===")
    
    try:
        # 학습 데이터 로드
        train_file = os.path.join(data_root, 'vad_nuscenes_infos_train.pkl')
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)
        
        # 맵 데이터 로드
        map_file = os.path.join(data_root, 'nuscenes_map_anns_val.json')
        with open(map_file, 'r') as f:
            map_data = json.load(f)
        
        # 일관성 확인
        print(f"✅ 학습 샘플 수: {len(train_data)}")
        print(f"✅ 맵 GT 수: {len(map_data['GTs'])}")
        
        # 토큰 일치 확인
        train_tokens = set(sample['token'] for sample in train_data)
        map_tokens = set(map_data['GTs'].keys())
        
        common_tokens = train_tokens & map_tokens
        print(f"✅ 공통 토큰 수: {len(common_tokens)}")
        
        if len(common_tokens) == len(train_tokens):
            print("✅ 모든 학습 샘플에 맵 GT가 있습니다")
        else:
            print(f"⚠️  일부 샘플에 맵 GT가 없습니다: {len(train_tokens) - len(common_tokens)}")
        
        # 이미지 파일 일치 확인
        missing_images = 0
        for sample in train_data:
            bev_image_path = os.path.join(data_root, 'bev_images', 'train', f"{sample['token']}.png")
            if not os.path.exists(bev_image_path):
                missing_images += 1
        
        if missing_images == 0:
            print("✅ 모든 샘플에 BEV 이미지가 있습니다")
        else:
            print(f"❌ {missing_images}개 샘플에 BEV 이미지가 없습니다")
        
        return missing_images == 0
        
    except Exception as e:
        print(f"❌ 데이터 일관성 테스트 실패: {str(e)}")
        return False


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='간단한 BEV 데이터셋 테스트')
    parser.add_argument('--data-root', default='data/nuscenes_dummy',
                       help='데이터 루트 경로')
    
    args = parser.parse_args()
    
    print(f"🚀 간단한 BEV 데이터셋 테스트 시작")
    print(f"데이터 경로: {args.data_root}")
    
    # 테스트 실행
    tests = [
        ("더미 데이터 로딩", test_dummy_data_loading),
        ("BEV 이미지 처리", test_bev_image_processing),
        ("GT 데이터 처리", test_gt_data_processing),
        ("데이터 일관성", test_data_consistency),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func(args.data_root)
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 실행 중 오류: {str(e)}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "="*50)
    print("📊 테스트 결과 요약")
    print("="*50)
    
    for test_name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{test_name}: {status}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n🎉 모든 테스트 통과! 더미 데이터셋이 정상적으로 생성되었습니다.")
        print("다음 단계:")
        print("1. mmdet3d 환경 설정 완료")
        print("2. 전체 파이프라인 테스트")
        print("3. 실제 학습 시작")
    else:
        print("\n❌ 일부 테스트 실패. 데이터 생성을 다시 확인해주세요.")
    
    print("\n테스트 완료!")


if __name__ == '__main__':
    main() 