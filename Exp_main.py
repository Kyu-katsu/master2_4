import Exp_Env as env
import Exp_IMM as imm
import Exp_RAF as raf
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
# pygame에서 여러 동적 객체들이 자유 운동을 하는 환경.
# 객체들의 속도에는 범위가 존재하며, env가 호출될때마다 한번의 time step이라고 생각하면 됨.
# input: 객체 개수, 시뮬레이션 step 설정
# output: 모든 동적 객체들의 위치 및 속도.


def main():
    # 초기 환경 설정 및 Pygame 초기화
    env.init_settings()
    screen, clock = env.init_pygame()

    num_objects = 3
    max_steps = 100  # real time step 진행
    real_dt = 0.1  # real time step

    # 객체 생성 (ID: 0, 1, 2)
    objects = [env.DynamicObject(id=i) for i in range(num_objects)]

    # 결과 저장: NumPy 배열 (행: 객체, 열: real time step)
    offsets = np.zeros((num_objects, max_steps))
    center_vels = np.zeros((num_objects, max_steps))
    pred_offsets = np.zeros((num_objects, max_steps))
    pred_center_vels = np.zeros((num_objects, max_steps))
    curr_Risks = np.zeros((num_objects, max_steps))
    pred_Risks = np.zeros((num_objects, max_steps))

    ### IMM Algorithm initialize
    initial_probs = [1/3, 1/3, 1/3]
    init_state_estimate = 6
    initial_variances = [1, 1, 1]
    IMMAlg = imm.IMM(initial_probs, init_state_estimate, initial_variances)

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
                    ### Pygame Environment One step Run
                    off, cent_vel = env.simulate_onestep(objects, real_dt, screen)
                    for obj in objects:
                        offsets[obj.id, step] = off[obj.id]
                        center_vels[obj.id, step] = cent_vel[obj.id]
                    print(f"Real time step {step}:")
                    for i in range(num_objects):
                        print(f"  Object {i}: Offset = {off[i]:.2f} m, Center Vel = {cent_vel[i]:.2f} m/s")


                    ### IMM Algorithm for Predicted pos, vel
                    for obj in objects:
                        pred_offset, pred_center_vel = imm.simulate_steps(obj, step, offsets, center_vels, IMMAlg)
                        pred_offsets[obj.id, step], pred_center_vels[obj.id, step] = pred_offset, pred_center_vel
                    for i in range(num_objects):
                        print(f"  현재 time step:{step}, Object {i}: Offset = {off[i]:.2f} m, Center Vel = {cent_vel[i]:.2f} m/s")
                        print(f"  미래 time step:{step+1}, Object {i}: Offset = {pred_offsets[i, step]:.2f} m, Center Vel = {pred_center_vels[i, step]:.2f} m/s")


                    ### Risk Assessment Function Check
                    for obj in objects:
                        curr_Risk = raf.cal_threat(off[obj.id], cent_vel[obj.id])
                        pred_Risk = raf.cal_threat(pred_offsets[obj.id, step], pred_center_vels[obj.id, step])
                        curr_Risks[obj.id, step] = curr_Risk
                        pred_Risks[obj.id, step] = pred_Risk


                    step += 1
                else:
                    running = False
        clock.tick(60)
    pygame.quit()
    print("\nFinal Results:")
    print("Offsets (m):")
    print(offsets)
    print("Pred Offsets (m):")
    print(pred_offsets)
    print("Center Velocities (m/s):")
    print(center_vels)
    print("Pred Center Velocities (m/s):")
    print(pred_center_vels)

    np.set_printoptions(suppress=True, formatter={'float_kind': lambda x: f"{x:.0f}"})
    print("Curr Risk Assessment :")
    print(curr_Risks)
    print("Pred Risk Assessment :")
    print(pred_Risks)


if __name__ == "__main__":
    main()
# if __name__=="__main__":
#     # num_objects = 3
#     #
#     # environment_init(num_objects)
#     #
#     # imm_init()
#     # #imm_init(single or multi = 0 or 1)
#     #
#     #     for
#     #         q_k, dot_q_k = iter_env()
#     #         # [객체 수, q_k, dot_q_k, time_step]
#     #
#     #         next_hat_q, next_hat_dot_q, next_hat_P = iter_imm(q_k, dot_q_k)
#     #
#     #         W = raf(q_k, dot_q_k)
#     #         W_prime = raf(next_hat_q, next_hat_dot_q)
#
