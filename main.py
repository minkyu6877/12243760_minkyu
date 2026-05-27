import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

def residuals(p, BS_positions, d_hat_u):
    """(내부 함수) 탐색 중인 좌표 p와 기지국 간의 거리 오차 계산"""
    distances = np.sqrt((BS_positions[0, :] - p[0])**2 + (BS_positions[1, :] - p[1])**2)
    return distances - d_hat_u

def your_algorithm(d_hat_u, BS_positions):
    """
    본인 알고리즘 작성 (1단계: Huber Loss 비선형 최적화)
    d_hat_u: 특정 사용자 1명의 18개 RTT 측정값 (18,)
    BS_positions: 18개 기지국의 좌표 (2, 18)
    """
    # 1. 초기 탐색 시작점 (기지국 무게중심)
    p0 = np.array([np.mean(BS_positions[0, :]), np.mean(BS_positions[1, :])])
    
    # 2. Huber Loss 최적화 수행 (NLOS 아웃라이어 무시 효과)
    res = least_squares(residuals, p0, loss='huber', f_scale=1.0, args=(BS_positions, d_hat_u))
    
    # 3. 최적화로 찾은 위치 결과 반환
    return res.x

def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path = 'DH_FR1.mat'
    data = sio.loadmat(mat_path, squeeze_me=False)
    
    # 업데이트된 가이드라인 반영: BS_positions로 통일하여 로드
    BS_positions = np.asarray(data['BS_positions'], dtype=float)      # (2, 18)
    d_hat = np.asarray(data['d_hat'], dtype=float)                    # (18, num_user)
    
    # ⚠️ 주의: Hidden Test Set에서는 정답지 'p'가 존재하지 않을 수 있으므로 
    # 평가용인 main.py에서는 아예 로드하지 않는 것이 가장 안전합니다.

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    
    for u in range(num_user):
        # your_algorithm 에 1명 분의 데이터(d_hat[:, u])를 올바른 문법으로 전달
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat

if __name__ == "__main__":
    # 코드가 정상 작동하는지 테스트용 출력 (제출 시 문제없음)
    result = main()
    print(f"✅ 위치 추정 완료! 반환된 배열 형태: {result.shape}")
