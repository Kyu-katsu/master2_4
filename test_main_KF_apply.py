import test_Env as env
import test_KF as kfmod
import test_RAF as raf
import test_graph as graph
import test_save_logs as save_logs
import test_MDP as mdp

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import pygame
import os


def main(iter):
    # 초기 환경 설정 및 Pygame 초기화
    env.init_settings()
    screen, clock = env.init_pygame()

    # 파라미터
    num_objects = 3
    save_steps = 100    # 총 time steps
    real_dt = 0.1    # 실제 시뮬 dt
    n_steps_pred = 5      # KF 예측 horizon

    # 객체 생성 (ID: 0,1,2)
    objects = [env.DynamicObject(id=i, dt=real_dt) for i in range(num_objects)]

    # 결과 저장 배열 생성
    offsets = np.zeros((num_objects, save_steps, n_steps_pred+1))
    center_vels = np.zeros((num_objects, save_steps, n_steps_pred+1))
    risk_vals = np.zeros((num_objects, save_steps, n_steps_pred+1))

    # --- Kalman Filter 초기화 ---
    # 각 객체별 CV Kalman Filter 인스턴스
    KFAlg = []
    for _ in range(num_objects):
        kf = kfmod.KalmanFilterCV(
            dt=real_dt,
            q_var=0.01,    # 프로세스 노이즈 분산 (튜닝 필요)
            r_var=0.05     # 측정 노이즈 분산 (튜닝 필요)
        )
        KFAlg.append(kf)

    step = 0
    running = True
    print("자동 시뮬레이션을 시작합니다.")

    while running and step < save_steps:
        # 종료 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # real time step 시뮬레이션
        off, cent_vel = env.simulate_onestep(objects, real_dt, screen)
        # 현재 offset/속도 저장
        for obj in objects:
            offsets[obj.id, step, 0] = off[obj.id]
            center_vels[obj.id, step, 0] = cent_vel[obj.id]

        # KF 기반 n-step 예측
        for obj in objects:
            pred_off, pred_vel = kfmod.simulate_n_steps_kf(
                obj_id      = obj.id,
                step        = step,
                offsets     = offsets,
                center_vels = center_vels,
                kf_filter   = KFAlg[obj.id],
                n_steps     = n_steps_pred
            )
            offsets[obj.id, step, 1:1+n_steps_pred] = pred_off
            center_vels[obj.id, step, 1:1+n_steps_pred] = pred_vel

        # 위험도 계산 (현재 + 예측)
        for obj in objects:
            for k in range(n_steps_pred+1):
                risk_vals[obj.id, step, k] = raf.cal_threat(
                    offsets[obj.id, step, k],
                    center_vels[obj.id, step, k]
                )

        step += 1
        clock.tick(60)

    pygame.quit()

    # --- MDP Action & Reward 계산 ---
    curr_Risks = risk_vals[:, :, 0]  # shape (num_objects, save_steps)
    # 가중치 감쇠 계수
    gamma   = 0.9
    weights = np.array([gamma**(i+1) for i in range(n_steps_pred)])
    # 예측 리스크: 가중합
    pred_Risks = np.sum(
        risk_vals[:,:,1:]*weights[np.newaxis,np.newaxis,:],
        axis=2
    )

    # 4가지 정책
    random_actions = mdp.random_action(num_objects, save_steps)
    random_reward  = mdp.cal_reward_after_10(num_objects, save_steps, curr_Risks, random_actions)

    curr_actions = mdp.cal_action_kyu(num_objects, save_steps, curr_Risks)
    curr_reward  = mdp.cal_reward_after_10(num_objects, save_steps, curr_Risks, curr_actions)

    pred_actions = mdp.cal_action_kyu(num_objects, save_steps, pred_Risks)
    pred_reward  = mdp.cal_reward_after_10(num_objects, save_steps, curr_Risks, pred_actions)

    total_actions = mdp.cal_total_based_action(num_objects, save_steps, curr_Risks, pred_Risks)
    total_reward  = mdp.cal_reward_after_10(num_objects, save_steps, curr_Risks, total_actions)

    # --- 그래프 및 로그 저장 ---
    os.makedirs("Offset_plots", exist_ok=True)
    filename = f"KF_{save_steps}step_pred{n_steps_pred}step_offset_plot_{iter}.png"
    graph.save_plot_offsets(save_steps, offsets, n_steps_pred, filename)

    os.makedirs("Logs", exist_ok=True)
    save_logs.save_risks(iter, save_steps, n_steps_pred, curr_Risks, pred_Risks)
    save_logs.save_actions(iter, save_steps, n_steps_pred,
                            random_actions, curr_actions, pred_actions, total_actions)
    save_logs.save_rewards(iter, save_steps, n_steps_pred,
                            random_reward, curr_reward, pred_reward, total_reward)
    save_logs.fit_exal(iter, save_steps, n_steps_pred)

    # 우승 정책 카운트 및 차이 반환
    rewards = [random_reward, curr_reward, pred_reward, total_reward]
    min_val = min(rewards)
    winners = [i for i, val in enumerate(rewards) if val==min_val]
    residual = curr_reward - total_reward
    return winners, residual


if __name__ == "__main__":
    iterations = 1
    win_count = np.zeros(15)
    max_residual = 0

    # 의미 있는 인덱스 조합 정의
    index_meanings = [
        [0], [1], [2], [3],
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
        [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]
    ]

    # 조합을 튜플로 바꿔서 딕셔너리로 매핑
    index_map = {tuple(k): i for i, k in enumerate(index_meanings)}

    # 메인 반복문
    for iter in range(iterations):
        min_indices, residual = main(iter)
        if residual > max_residual:
            max_residual = residual
            max_residual_iter = iter
        key = tuple(sorted(min_indices))
        if key in index_map:
            win_count[index_map[key]] += 1

    # 출력
    for i, count in enumerate(win_count):
        if count != 0:
            print(f"{index_meanings[i]} 갯수 : {count}")
