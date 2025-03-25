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

def environment_init(num_objects):
    env.init_settings()
    screen, clock = env.init_pygame()

    objects = [env.DynamicObject(id=i) for i in range(num_objects)]



if __name__=="__main__":
    # num_objects = 3
    #
    # environment_init(num_objects)
    #
    # imm_init()
    # #imm_init(single or multi = 0 or 1)
    #
    #     for
    #         q_k, dot_q_k = iter_env()
    #         # [객체 수, q_k, dot_q_k, time_step]
    #
    #         next_hat_q, next_hat_dot_q, next_hat_P = iter_imm(q_k, dot_q_k)
    #
    #         W = raf(q_k, dot_q_k)
    #         W_prime = raf(next_hat_q, next_hat_dot_q)

    ### Env initialize
    env.init_settings()
    screen, clock = env.init_pygame()

    # 객체 생성: 실제 공간 좌표를 기반으로 초기화된 동적 객체들 (ID는 0 부터 할당)
    num_objects = 3
    objects = [env.DynamicObject(id=i) for i in range(num_objects)]

    # 시뮬레이션 step 설정
    time_steps = 100

    # 위협 평가값 저장 리스트
    threat_values = np.zeros((num_objects, time_steps))

    ### IMM initialize
    num_ROI = 3 #구역 수
    pos_values = np.zeros((num_objects, time_steps))
    predicted_loc_values = np.zeros((num_objects, time_steps+1))
    predicted_cov_values = np.zeros((num_objects, time_steps+1))
    for i in range(num_objects):
        predicted_cov_values[i][0] = 0
    # 모델 확률 초기화
    model_probs = np.zeros((num_objects, num_ROI))
    for i in range(num_objects):
        for j in range(num_ROI):
            model_probs[i][j] = 1/3
    # 객체 별 Offset 초기화
    position = np.zeros((num_objects))
    # 객체 별 Offset 속도 초기화
    velocity_values = np.zeros((num_objects))


    ### Env 실행 및 IMM 실행
    for steps in range(time_steps):
        screen.fill(env.WHITE)
        # 동심원(구역) 그리기: 실제 공간을 기준으로 색칠된 구역을 화면에 표시
        env.draw_circles(screen)

        # 각 객체에 대해: 화면에 그리기, offset 및 속도 화살표 표현, 그리고 실제 좌표 업데이트
        for obj in objects:
            # 디버그 출력 (콘솔에 실제 좌표와 속도 출력)
            print(f"Object ID: {obj.id}, lateral offset: ({obj.offset:.2f}) Actual Pos: ({obj.actual_x:.2f}, {obj.actual_y:.2f}), Speed: {obj.speed:.2f}")
            obj.draw(screen)
            env.draw_offset(screen, obj)
            env.draw_velocity_arrows(screen, obj)

            position[obj.id] = obj.offset
            pos_values[obj.id][steps] = position[obj.id]
            velocity_values[obj.id] = obj.radial_speed

            obj.update(0.1)





        pygame.display.flip()

    pygame.quit()
