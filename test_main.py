import test_Env as env
import test_IMM as imm
import test_RAF as raf
import test_graph as graph
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import pygame
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


# class imm::
# input: 현재 위치(q_k), 현재 속도(dot_q_k)
# output: 미래 위치(hat_q_(k+1)), 미래 속도(hat_dot_q_(k+1)=dot_q_k)

# class raf::
# input: k시점 위치(q_k), k시점 속도(dot_q_k)
# output: k시점 위협 위험도(W_k)

# class env::
# pygame에서 여러 동적 객체들이 자유 운동을   하는 환경.
# 객체들의 속도에는 범위가 존재하며, env가 호출될때마다 한번의 time step이라고 생각하면 됨.
# input: 객체 개수, 시뮬레이션 step 설정
# output: 모든 동적 객체들의 위치 및 속도.


def    main():
    # 초기 환경 설정 및 Pygame 초기화
    env.init_settings()
    screen, clock = env.init_pygame()

    num_objects = 3
    max_steps = 100     # 총 100 real time steps (예: 0.1초씩이면 10초)
    real_dt = 1    # real time step 간격 (초)
    n_steps_pred = 1   # IMM 예측 horizon (n time steps, 예: 1 또는 5)

    # 객체 생성 (ID: 0, 1, 2)
    objects = [env.DynamicObject(id=i) for i in range(num_objects)]

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

    # (진우 수정) 각 객체들이 따로 따로 imm 클래스를 가지고 있게 설정.
    IMMAlg = []
    for i in range(num_objects):
        IMMAlg.append(imm.IMM(initial_probs, init_state_estimate, initial_variances, max_steps, n_steps=n_steps_pred))

    # 이렇게 하면 모든 객체가 같은 파라미터를 공유함.
    # IMMAlg = imm.IMM(initial_probs, init_state_estimate, initial_variances, max_steps, n_steps=n_steps_pred)

    step = 0
    running = True
    print("Press SPACE to advance one real time step ({} sec each).".format(real_dt))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 스페이스바 누르면 한 real time step 진행
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if step < max_steps:
                    ### Pygame Environment One 'real_dt' step Run
                    off, cent_vel = env.simulate_onestep(objects, real_dt, screen)
                    for obj in objects:
                        offsets[obj.id, step] = off[obj.id]
                        center_vels[obj.id, step] = cent_vel[obj.id]
                    print(f"Real time step {step}:")
                    for i in range(num_objects):
                        print(f"  Object {i}: Offset = {off[i]:.2f} m, Center Vel = {cent_vel[i]:.2f} m/s")


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
                        # 먼저, 각 예측 시점의 위험도를 계산하고 총합
                        pred_risk_array = []
                        for k in range(n_steps_pred):
                            r_val = pred_offsets[obj.id, step, k]
                            v_val = pred_center_vels[obj.id, step, k]
                            pred_risk_array.append(raf.cal_threat(r_val, v_val))
                        pred_Risk = np.sum(pred_risk_array)
                        curr_Risks[obj.id, step] = curr_Risk
                        pred_Risks[obj.id, step] = pred_Risk

                    for i in range(num_objects):
                        print(
                            f"  Object {i}: Current Risk = {curr_Risks[i, step]:.2f}, Predicted Risk = {pred_Risks[i, step]:.2f}")
                    step += 1
                else:
                    running = False
        clock.tick(60)
    pygame.quit()

    # (진우 수정) 액션, 리워드 계산
    import test_MDP as mdp
    curr_based_actions = mdp.cal_acrtion(num_objects, max_steps, curr_Risks)
    pred_based_actions = mdp.cal_acrtion(num_objects, max_steps, pred_Risks)

    curr_based_reward = mdp.cal_reward(num_objects, max_steps, curr_Risks, curr_based_actions)
    pred_based_reward = mdp.cal_reward(num_objects, max_steps, curr_Risks, pred_based_actions)


    print("\nFinal Results:")
    print("Offsets (m), {}:".format(offsets.shape))
    print(offsets)
    print("Predicted Offsets (m), {}:".format(pred_offsets.shape))
    print(pred_offsets)
    print("Center Velocities (m/s):")
    print(center_vels)
    print("Predicted Center Velocities (m/s):")
    print(pred_center_vels)
    print("Current Risk Assessments:")
    print(curr_Risks)
    print("Predicted Risk Assessments:")
    print(pred_Risks)

    # (진우 수정) 액션, 리워드 출력
    print("Current Risk Based Actions:")
    print(curr_based_actions)
    print("Pred Risk Based Actions:")
    print(pred_based_actions)
    print("Current Risk Based reward:")
    print(curr_based_reward)
    print("Pred Risk Based reward:")
    print(pred_based_reward)

    graph.plot_offsets(max_steps, offsets, pred_offsets, n_steps_pred)


if __name__ == "__main__":
    main()

