import test_Env as env
import test_IMM as imm
import test_RAF as raf
import test_graph as graph
import test_save_logs as save_logs

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import pygame
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import pandas as pd

# class imm::
# input: 현재 위치(q_k), 현재 속도(dot_q_k)
# output: 미래 위치(hat_q_(k+1)), 미래 속도(hat_dot_q_(k+1)=dot_q_k)

# class raf::
# input: k시점 위치(q_k), k시점 속도(dot_q_k)
# output: k시점 위협 위험도(W_k)

# class env::
# pygame에서 여러 동적 객체들이 자유 운동하는 환경.
# 객체들의 속도에는 범위가 존재하며, env가 호출될 때마다 한번의 time step이라고 생각하면 됨.
# input: 객체 개수, 시뮬레이션 step 설정
# output: 모든 동적 객체들의 위치 및 속도.

def main(iter):
    # 초기 환경 설정 및 Pygame 초기화
    env.init_settings()
    screen, clock = env.init_pygame()

    num_objects = 3
    max_steps = 100    # 총 100 real time steps (예: 0.1초씩이면 10초)
    real_dt = 0.1        # real time step 간격 (초)
    n_steps_pred = 3    # IMM 예측 horizon (n time steps, 예: 1 또는 5)

    # 객체 생성 (ID: 0, 1, 2)
    objects = [env.DynamicObject(id=i, dt = real_dt) for i in range(num_objects)]

    # 결과 저장 배열 (NumPy 배열; 행: 객체, 열: time step)
    offsets = np.zeros((num_objects, max_steps))
    center_vels = np.zeros((num_objects, max_steps))
    pred_offsets = np.zeros((num_objects, max_steps, n_steps_pred))
    pred_center_vels = np.zeros((num_objects, max_steps, n_steps_pred))
    curr_Risks = np.zeros((num_objects, max_steps))
    pred_Risks = np.zeros((num_objects, max_steps))

    ### IMM Algorithm initialize
    initial_probs = [1/3, 1/3, 1/3]
    init_state_estimate = 6
    initial_variances = [1, 1, 1]

    # 각 객체마다 개별 imm 인스턴스 생성
    IMMAlg = []
    for i in range(num_objects):
        IMMAlg.append(imm.IMM(initial_probs, init_state_estimate, initial_variances, max_steps, n_steps=n_steps_pred))

    step = 0
    running = True
    print("자동 시뮬레이션을 시작합니다.")

    while running and step < max_steps:
        # 이벤트 처리 (종료 이벤트 등)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 한 real time step 실행
        off, cent_vel = env.simulate_onestep(objects, real_dt, screen)
        for obj in objects:
            offsets[obj.id, step] = off[obj.id]
            center_vels[obj.id, step] = cent_vel[obj.id]
        # print(f"Real time step {step}:")
        # for i in range(num_objects):
        #     print(f"  Object {i}: Offset = {off[i]:.2f} m, Center Vel = {cent_vel[i]:.2f} m/s")

        ### IMM Algorithm for 'n_steps' Predicted pos, vel
        for obj in objects:
            pred_off, pred_cent_vel = imm.simulate_n_steps(obj, step, offsets, center_vels, IMMAlg[obj.id])
            # pred_off, pred_cent_vel은 1D 배열 길이 n_steps_pred
            pred_offsets[obj.id, step, :] = pred_off
            pred_center_vels[obj.id, step, :] = pred_cent_vel

        ### Risk Assessment Function Check / 'n_steps' 예측 위험도
        for obj in objects:
            curr_Risk = raf.cal_threat(off[obj.id], cent_vel[obj.id])
            # pred_Risk는 n_steps 예측에 대한 위험도 총합
            pred_risk_array = []
            for k in range(n_steps_pred):
                r_val = pred_offsets[obj.id, step, k]
                v_val = pred_center_vels[obj.id, step, k]
                pred_risk_array.append(raf.cal_threat(r_val, v_val))
            pred_Risk = np.sum(pred_risk_array)
            curr_Risks[obj.id, step] = curr_Risk
            pred_Risks[obj.id, step] = pred_Risk

        # for i in range(num_objects):
        #     print(f"  Object {i}: Current Risk = {curr_Risks[i, step]:.2f}, Predicted Risk = {pred_Risks[i, step]:.2f}")

        step += 1
        clock.tick(60)  # 초당 최대 60 프레임으로 실행

    pygame.quit()

    # 액션, 리워드 계산
    import test_MDP as mdp
    curr_based_actions = mdp.cal_action(num_objects, max_steps, curr_Risks)
    pred_based_actions = mdp.cal_action(num_objects, max_steps, pred_Risks)

    curr_based_reward = mdp.cal_reward_after_10(num_objects, max_steps, curr_Risks, curr_based_actions)
    pred_based_reward = mdp.cal_reward_after_10(num_objects, max_steps, curr_Risks, pred_based_actions)

    # (진우 수정2) Total risk 방식 액션, 리워드 계산
    total_based_actions = mdp.cal_total_based_action(num_objects, max_steps, curr_Risks, pred_Risks)
    total_based_reward = mdp.cal_reward_after_10(num_objects, max_steps, curr_Risks, total_based_actions)

    # (진우 수정3) Random policy 액션, 리워드 계산
    random_based_actions = mdp.random_action(num_objects, max_steps)
    random_based_reward = mdp.cal_reward_after_10(num_objects, max_steps, curr_Risks, random_based_actions)

    # print("\nFinal Results:")
    # print("Offsets (m), {}:".format(offsets.shape))
    # print(offsets)
    # print("Predicted Offsets (m), {}:".format(pred_offsets.shape))
    # print(pred_offsets)
    # print("Center Velocities (m/s):")
    # print(center_vels)
    # print("Predicted Center Velocities (m/s):")
    # print(pred_center_vels)
    # print("Current Risk Assessments:")
    # print(curr_Risks)
    # print("Predicted Risk Assessments:")
    # print(pred_Risks)

    # # (진우 수정2) Total Risk Based 액션, 리워드 출력 추가.
    # # (진우 수정3) Random policy 액션, 리워드 출력 추가
    # print("Random Risk Actions:")
    # print(random_based_actions)
    # print("Current Risk Based Actions:")
    # print(curr_based_actions)
    # print("Pred Risk Based Actions:")
    # print(pred_based_actions)
    # print("total Risk Based Actions:")
    # print(total_based_actions)
    # print("Random Risk Based reward:")
    # print(random_based_reward)
    # print("Current Risk Based reward:")
    # print(curr_based_reward)
    # print("Pred Risk Based reward:")
    # print(pred_based_reward)
    # print("total Risk Based reward:")
    # print(total_based_reward)

    # (진우 수정3) 그래프 저장. 저장 위치: Offset_plots 파일 안.
    os.makedirs("Offset_plots", exist_ok=True)
    filename = f"offset_plot_{iter}.png"
    graph.save_plot_offsets(max_steps, offsets, pred_offsets, n_steps_pred,filename)

    # (진우 수정3) 로그 저장. 저장 위치: Logs 파일 안.
    os.makedirs("Logs", exist_ok=True)
    save_logs.save_risks(iter, max_steps, curr_Risks, pred_Risks)
    save_logs.save_actions(iter, random_based_actions, curr_based_actions, pred_based_actions, total_based_actions)
    save_logs.save_rewards(iter, random_based_reward, curr_based_reward, pred_based_reward, total_based_reward)
    save_logs.fit_exal(iter)

    # (진우 수정4) 여러번 돌려서 case 1~4의 우승 횟수(리워드 제일 작은지) 확인
    rewards = [random_based_reward, curr_based_reward, pred_based_reward, total_based_reward]
    min_reward = min(rewards)
    min_indices = [i for i, val in enumerate(rewards) if val == min_reward]
    print(iter, min_indices)

    # (진우 수정4) 차이 큰 애들 볼려고
    residual = curr_based_reward - total_based_reward

    return min_indices, residual

if __name__ == "__main__":
    iteration = 10

    # (진우 수정4) 여러번 돌려서 case 1~4의 우승 횟수(리워드 제일 작은지) 확인
    win_count = np.zeros((15))
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
    for iter in range(iteration):
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

    print("max_residual_iter :", max_residual_iter)