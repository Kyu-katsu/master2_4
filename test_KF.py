import numpy as np

class KalmanFilterCV:
    def __init__(self, dt, q_var, r_var, init_P=None):
        self.dt = dt        # 0.1
        # 상태 전이 행렬
        self.F = np.array([[1, dt],
                           [0,  1]])
        # 관측 행렬 (위치만 측정)
        self.H = np.array([[1, 0]])
        # 프로세스 노이즈 공분산
        self.Q = q_var * np.array([[dt**2/2,    0],
                                   [   0,  dt**2/2]])
        # 측정 노이즈 공분산
        self.R = np.array([[r_var]])
        # 초기 상태 추정 (0으로 두고, 나중에 덮어쓰기)
        self.x = np.zeros((2, 1))
        # 초기 공분산
        self.P = init_P if init_P is not None else np.eye(2)

    def predict(self):
        """
        상태 예측 단계:
          x_{k|k-1} = F x_{k-1|k-1}
          P_{k|k-1} = F P_{k-1|k-1} F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        측정 업데이트 단계:
          y = z - H x_{k|k-1}
          S = H P_{k|k-1} H^T + R
          K = P_{k|k-1} H^T S^{-1}
          x_{k|k} = x_{k|k-1} + K y
          P_{k|k} = (I - K H) P_{k|k-1}
        z: 측정된 위치 (스칼라)
        """
        y = np.array([[z]]) - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

    def set_state(self, q, v, P=None):
        """
        외부에서 현재 측정된 상태로 초기화 (옵션지만 KF 유지하려면 매 스텝 update 권장)
        """
        self.x = np.array([[q],[v]])
        if P is not None:
            self.P = P

    def get_state(self):
        """(q, v) 튜플 반환"""
        return float(self.x[0]), float(self.x[1])


def simulate_n_steps_kf(obj_id, step, offsets, center_vels, kf_filter, n_steps):
    """
    KF 예측 함수. 1) 현재 측정치 업데이트 2) n-스텝 예측
    - obj_id        : 객체 ID
    - step          : 현재 시뮬레이션 스텝 인덱스
    - offsets       : shape (num_obj, total_steps, ?)
    - center_vels   : shape (num_obj, total_steps, ?)
    - kf_filter     : KalmanFilterCV 인스턴스
    - n_steps       : 예측 horizon
    반환:
      - pred_offsets    : 길이 n_steps, 예측 위치
      - pred_velocities : 길이 n_steps, 예측 속도
    """

    pred_offsets = np.zeros(n_steps)
    pred_center_vels = np.zeros(n_steps)

    # 초기값 설정: 현재 offset과 중심 속도를 시작 상태로 사용
    current_offset = offsets[obj_id, step, 0]
    current_velocity = center_vels[obj_id, step, 0]

    # (선택적) 이전 예측 결과를 유지하려면 set_state 호출 생략하고, 바로 update 만 수행 가능
    kf_filter.set_state(q0, v0)
    # 2) 실제 측정값을 반영한 업데이트
    kf_filter.update(q0)
    # 3) 순수 예측: 업데이트된 상태를 기반으로 n번 예측
    for k in range(n_steps):
        kf_filter.predict()
        q_pred, v_pred = kf_filter.get_state()
        pred_offsets[k] = q_pred
        pred_center_vels[k]    = v_pred

    return pred_offsets, pred_center_vels
