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
    max_steps = 10  # 총 10 real time step 진행
    real_dt = 1.0  # real time step: 1초 (실제 시간)

    # 객체 생성 (ID: 0, 1, 2)
    objects = [env.DynamicObject(id=i) for i in range(num_objects)]

    # 결과 저장: NumPy 배열 (행: 객체, 열: real time step)
    offsets = np.zeros((num_objects, max_steps))
    center_vels = np.zeros((num_objects, max_steps))

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
                    off, cent_vel = env.simulate_onestep(objects, real_dt, screen)
                    for obj in objects:
                        offsets[obj.id, step] = off[obj.id]
                        center_vels[obj.id, step] = cent_vel[obj.id]
                    print(f"Real time step {step}:")
                    for i in range(num_objects):
                        print(f"  Object {i}: Offset = {off[i]:.2f} m, Center Vel = {cent_vel[i]:.2f} m/s")
                    step += 1
                else:
                    running = False
        clock.tick(60)
    pygame.quit()
    print("\nFinal Results:")
    print("Offsets (m):")
    print(offsets)
    print("Center Velocities (m/s):")
    print(center_vels)


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
