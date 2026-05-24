import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

def residuals(p, p_bs, d_hat_u):
    """(내부 함수) 탐색 중인 좌표 p와 기지국 간의 거리 오차 계산"""
    distances = np.sqrt((p_bs[0, :] - p[0])**2 + (p_bs[1, :] - p[1])**2)
    return distances - d_hat_u

def your_algorithm(d_hat_u, p_bs):
    """
    본인 알고리즘 작성 (1단계: Huber Loss 비선형 최적화)
    d_hat_u: 특정 사용자 1명의 18개 RTT 측정값 (18,)
    p_bs: 18개 기지국의 좌표 (2, 18)
    """
    # 1. 초기 탐색 시작점 (기지국 무게중심)
    p0 = np.array([np.mean(p_bs[0, :]), np.mean(p_bs[1, :])])
    
    # 2. Huber Loss 최적화 수행 (NLOS 아웃라이어 무시 효과)
    res = least_squares(residuals, p0, loss='huber', f_scale=1.0, args=(p_bs, d_hat_u))
    
    # 3. 최적화로 찾은 위치 결과 반환
    return res.x

def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path = 'DH_FR1.mat'
    data = sio.loadmat(mat_path, squeeze_me=False)
    
    # [안전 장치] 가이드라인은 p_bs, 실제 데이터는 BS_positions인 함정 대비
    if 'p_bs' in data:
        p_bs = np.asarray(data['p_bs'], dtype=float)      # (2, 18)
    else:
        p_bs = np.asarray(data['BS_positions'], dtype=float)
        
    d_hat = np.asarray(data['d_hat'], dtype=float)        # (18, num_user)
    
    # ⚠️ 주의: Hidden Test Set에서는 정답지 'p'가 존재하지 않을 수 있으므로 
    # 평가용인 main.py에서는 아예 로드하지 않는 것이 가장 안전합니다.

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    
    for u in range(num_user):
        # your_algorithm 에 1명 분의 데이터(d_hat[:, u])를 올바른 문법으로 전달
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat

if __name__ == "__main__":
    # 코드가 정상 작동하는지 테스트용 출력
    result = main()
    print(f"✅ 위치 추정 완료! 반환된 배열 형태: {result.shape}")
