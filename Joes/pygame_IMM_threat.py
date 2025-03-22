
import time
from tkinter.constants import CENTER

import pygame
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.stats import norm
import IMM_multi_object as IMM
import threat_assessment as Threat
import time

# 초기 설정
def init_settings():
    global WIDTH, HEIGHT, SPACE_SIZE, SCALE, WHITE, RED, GREEN, BLUE, BLACK, CENTER_X, CENTER_Y
    WIDTH, HEIGHT = 1000, 1000
    SPACE_SIZE = 25
    SCALE = WIDTH // SPACE_SIZE
    CENTER_X, CENTER_Y = 12.5, 12.5

    # 색상 정의
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)

# Pygame 초기화 및 화면 설정
def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    return screen, clock


# 동적 객체 클래스 정의
class DynamicObject:
    def __init__(self, id):

        self.id = id  # 객체 ID 추가
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.r = np.random.uniform(2, 10)
        self.dx = self.r * np.cos(self.direction)
        self.dy = self.r * np.sin(self.direction)
        self.x = CENTER_X + self.dx
        self.y = CENTER_Y + self.dy
        self.speed = np.random.uniform(0, 1)
        self.speed_x = self.speed * np.cos(self.direction)
        self.speed_y = self.speed * np.sin(self.direction)

        self.radial_speed = 0

        self.ax = np.random.uniform(-0.5, 0.5)
        self.ay = np.random.uniform(-0.5, 0.5)

    def update(self, dt):
        # 가속도 업데이트
        self.ax = np.clip(self.ax + np.random.uniform(-0.05, 0.05), -0.5, 0.5)
        self.ay = np.clip(self.ay + np.random.uniform(-0.05, 0.05), -0.5, 0.5)

        # 속도 업데이트
        self.speed_x = self.speed_x + self.ax * dt
        self.speed_y = self.speed_y + self.ay * dt

        # 방향 계산: 기존 방향 + 랜덤 변동
        target_direction = math.atan2(self.speed_y, self.speed_x)
        self.direction = 0.9 * self.direction + 0.1 * target_direction  # 점진적 변경
        self.direction += np.random.normal(0, 0.05)  # 작은 랜덤 변화 추가

        # 속력 계산: 제한 범위 설정
        self.speed = np.sqrt(self.speed_x**2 + self.speed_y**2)
        self.speed = np.clip(self.speed, 0, 1)

        # 위치 업데이트
        self.x += self.speed * np.cos(self.direction) * dt
        self.y += self.speed * np.sin(self.direction) * dt

        # 중심(CENTER_X, CENTER_Y)으로부터의 상대 위치
        self.dx = self.x - CENTER_X
        self.dy = self.y - CENTER_Y

        ########## Offset 속도 계산부분 ##############
        self.radial_speed = self.r - math.sqrt(self.dx ** 2 + self.dy ** 2)
        self.r = math.sqrt(self.dx ** 2 + self.dy ** 2)


        ### 2~10 경계선 ###
        boundary_max_radius = 10
        boundary_min_radius = 2
        # 10 보다 클때 경계선
        if self.r > boundary_max_radius:
            # 법선 벡터 계산
            nx = self.dx / self.r
            ny = self.dy / self.r

            # 현재 속도 벡터
            vx = self.speed * np.cos(self.direction)
            vy = self.speed * np.sin(self.direction)

            # 내적 계산
            dot = vx * nx + vy * ny

            # 반사된 속도 벡터 계산
            vx_reflected = vx - 2 * dot * nx
            vy_reflected = vy - 2 * dot * ny

            # 새로운 방향 설정 (반사된 속도 벡터 기준)
            self.direction = math.atan2(vy_reflected, vx_reflected)

            # (옵션) elf.x = CENTER_X + nx * boundary_max_radius
            self.y = CENTER_Y + ny * boundary_max_radius
        # 2보다 작을때 경계선
        if self.r < boundary_min_radius:
            # 법선 벡터 계산
            nx = self.dx / self.r
            ny = self.dy / self.r

            # 현재 속도 벡터
            vx = self.speed * np.cos(self.direction)
            vy = self.speed * np.sin(self.direction)

            # 내적 계산
            dot = vx * nx + vy * ny

            # 반사된 속도 벡터 계산
            vx_reflected = vx - 2 * dot * nx
            vy_reflected = vy - 2 * dot * ny

            # 새로운 방향 설정 (반사된 속도 벡터 기준)
            self.direction = math.atan2(vy_reflected, vx_reflected)

            # (옵션) 객체를 경계 위로 재배치: 경계를 약간 벗어나지 않도록
            self.x = CENTER_X + nx * boundary_min_radius
            self.y = CENTER_Y + ny * boundary_min_radius




    def draw(self, screen):
        # 객체 원 그리기
        pygame.draw.circle(screen, RED, (int(self.x * SCALE), int(self.y * SCALE)), 5)

        # 객체 번호 표시
        font = pygame.font.Font(None, 24)
        text = font.render(str(self.id), True, (0, 0, 0))
        screen.blit(text, (int(self.x * SCALE) + 10, int(self.y * SCALE) - 10))


    @property
    def vx(self):
        return self.speed * np.cos(self.direction)

    @property
    def vy(self):
        return self.speed * np.sin(self.direction)


# 동심원 그리기 함수
def draw_circles(screen):
    pygame.draw.circle(screen, BLUE, (WIDTH // 2, HEIGHT // 2), 12 * SCALE, 3)   # RoI 3
    pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 10 * SCALE, 1)  # RoI 3 중심선
    pygame.draw.circle(screen, GREEN, (WIDTH // 2, HEIGHT // 2), 8 * SCALE, 3)  # RoI 2
    pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 6 * SCALE, 1)  # RoI 2 중심선
    pygame.draw.circle(screen, RED, (WIDTH // 2, HEIGHT // 2), 4 * SCALE, 3)    # RoI 1
    pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 2 * SCALE, 1)   # RoI 1 중심선


# 오프셋 계산 및 표시 함수
def draw_offset(screen, obj):
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    offset = math.sqrt((obj.x * SCALE - center_x) ** 2 + (obj.y * SCALE - center_y) ** 2)
    font = pygame.font.Font(None, 24)
    text = font.render(f"Offset: {offset / SCALE:.2f}", True, (0, 0, 0))
    screen.blit(text, (obj.x * SCALE, obj.y * SCALE - 20))
    # return offset / SCALE


def cal_offset(obj):
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    offset = math.sqrt((obj.x * SCALE - center_x) ** 2 + (obj.y * SCALE - center_y) ** 2)
    RoI_centerline_offset = (round(abs((offset / SCALE) - 5), 2), round(abs((offset / SCALE) - 20), 2), round(abs((offset / SCALE) - 40), 2))  # RoI별 거리
    return RoI_centerline_offset


def draw_velocity_arrows(screen, obj, center_x, center_y):
    start_pos = (int(obj.x * SCALE), int(obj.y * SCALE))  # 객체 위치

    # 화살표를 그리는 내부 함수
    def draw_arrow(color, start, vector, scale=30):
        magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)  # 속도 크기
        if magnitude == 0:
            return  # 속도가 0이면 화살표 없음
        unit_vector = (vector[0] / magnitude, vector[1] / magnitude)  # 방향 벡터 계산
        end_pos = (start[0] + int(unit_vector[0] * magnitude * scale),
                   start[1] + int(unit_vector[1] * magnitude * scale))
        pygame.draw.line(screen, color, start, end_pos, 2)  # 화살표 몸체

        # 화살표 촉 계산
        arrow_size = 10  # 촉 크기
        arrow_point1 = (end_pos[0] - int(unit_vector[0] * arrow_size - unit_vector[1] * arrow_size // 2),
                        end_pos[1] - int(unit_vector[1] * arrow_size + unit_vector[0] * arrow_size // 2))
        arrow_point2 = (end_pos[0] - int(unit_vector[0] * arrow_size + unit_vector[1] * arrow_size // 2),
                        end_pos[1] - int(unit_vector[1] * arrow_size - unit_vector[0] * arrow_size // 2))
        pygame.draw.polygon(screen, color, [end_pos, arrow_point1, arrow_point2])

    # x축 속도 화살표 (녹색)
    draw_arrow(GREEN, start_pos, (obj.vx, 0))

    # y축 속도 화살표 (파란색)
    draw_arrow(BLUE, start_pos, (0, obj.vy))

    # 중심점 방향 속도 화살표 (빨간색)
    # 중심점과 객체 간 방향 벡터 계산
    vcx = center_x / SCALE - obj.x  # 중심점으로의 x축 방향
    vcy = center_y / SCALE - obj.y  # 중심점으로의 y축 방향
    center_magnitude = math.sqrt(vcx ** 2 + vcy ** 2)  # 중심점 방향 벡터 크기
    if center_magnitude > 0:
        vcx /= center_magnitude  # 단위 벡터화
        vcy /= center_magnitude

    # 중심점 방향 속도의 크기를 기반으로 화살표 길이를 조정
    center_speed = obj.vx * vcx + obj.vy * vcy  # 중심점 방향 속도 크기 (스칼라 곱)
    draw_arrow(RED, start_pos, (vcx * center_speed, vcy * center_speed))


def calculate_center_velocity(obj, center_x, center_y):
    # 중심점 벡터
    vcx = center_x / SCALE - obj.x
    vcy = center_y / SCALE - obj.y
    center_vector = np.array([vcx, vcy])

    # 객체 속도 벡터
    velocity_vector = np.array([obj.vx, obj.vy])

    # 중심점 방향 속도 계산
    center_vector_magnitude = np.linalg.norm(center_vector)
    if center_vector_magnitude == 0:  # 중심점과 겹치면 속도 0
        return 0

    center_velocity = np.dot(velocity_vector, center_vector) / center_vector_magnitude
    return round(center_velocity, 2)




def main_timesteps(time_steps):
    init_settings()
    screen, clock = init_pygame()

    # 객체 생성 (ID 부여)
    num_objects = 3  # 다중 객체 수
    num_ROI = 3 #구역 수
    objects = [DynamicObject(id=i + 1) for i in range(num_objects)]   # 객체 마릿수 지정

    threat_values = np.zeros((num_objects, time_steps))

    ### IMM 초기화 부분 ###
    pos_values = np.zeros((num_objects, time_steps))
    predicted_loc_values = np.zeros((num_objects, time_steps+1))

    # 모델 확률 초기화
    initial_probs = np.zeros((num_objects, num_ROI))
    for i in range(num_objects):
        for j in range(num_ROI):
            initial_probs[i][j] = 1/3

    # 모델 분산 초기화
    initial_variances = np.zeros((num_objects, num_ROI))
    for i in range(num_objects):
        for j in range(num_ROI):
            initial_variances[i][j] = 1

    # 객체 별 Offset 초기화
    position = np.zeros((num_objects))
    # 객체 별 Offset 속도 초기화
    velocity_values = np.zeros((num_objects))

    ### 객체 수만큼 IMM 생성 ###
    imm = []
    for i in range(num_objects):
        imm.append(IMM.IMM(i+1 ,initial_probs[i], position[i], initial_variances[i], time_steps))

    # 객체별 예측 Offset 초기화
    predicted_loc = np.zeros(num_objects)
    for i in range(num_objects):
        predicted_loc[i] = position[i]


    #################### IMM 실행 ###########################
    for steps in range(time_steps):
        screen.fill(WHITE)
        # 동심원 그리기
        draw_circles(screen)
        # 중심점 설정
        center_x, center_y = WIDTH // 2, HEIGHT // 2

        ### 각 스텝마다 개체별로 위치, 속도 저장 ###
        for i in range(num_objects):
            position[i] = objects[i].r
        for obj in range(num_objects):
            objects[obj].draw(screen)
            velocity_values[obj] = objects[obj].radial_speed

        ### 위협 평가 ###
        for obj in range(num_objects):
            if position[obj]<=2:
                position[obj] = 2.001
            threat_values[obj][steps] = Threat.cal_threat(12, position[obj], velocity_values[obj])
        ###############

        print("=============================================================")
        print("{}번째 IMM 진행".format(steps+1))


        TPM = np.zeros((num_objects, num_ROI, num_ROI))
        mixed_P = np.zeros((num_objects, num_ROI))
        mixed_bar_q = np.zeros((num_objects, num_ROI))
        mixed_mu = np.zeros((num_objects, num_ROI))
        mixed_ratio = np.zeros((num_objects, num_ROI, num_ROI))
        residual_term = np.zeros((num_objects, num_ROI))
        filtered_mu = np.zeros((num_objects, num_ROI))
        pos_residual = np.zeros(num_objects)

        for model_i in range(num_objects):
            print("===={}번째 객체====:".format(model_i+1))

            print("위치 :", position[model_i], ",속도 :", velocity_values[model_i])

            TPM[model_i] = imm[model_i].generate_TPM_CDF(velocity_values[model_i])
            mixed_mu[model_i], mixed_ratio[model_i], mixed_bar_q[model_i], mixed_P[model_i] \
                = imm[model_i].mixed_prediction(TPM[model_i])  # 예측 단계

            residual_term[model_i] = position[model_i] - mixed_bar_q[model_i]
            filtered_mu[model_i] = imm[model_i].filter_prediction(mixed_mu[model_i], mixed_P[model_i],
                                                                  residual_term[model_i])  # 필터 단계, \mu만 갱신
            imm[model_i].mu_values[steps] = filtered_mu[model_i]
            print("Filtered mu: {}".format(filtered_mu[model_i]))

            pos_residual[model_i] = predicted_loc[model_i] - position[model_i]
            predicted_loc[model_i] = filtered_mu[model_i][0] * 2 + filtered_mu[model_i][1] * 6 + filtered_mu[model_i][2] * 10
            ("Predict loc: {}".format(predicted_loc[model_i]))

            predicted_loc_values[model_i][steps + 1] = predicted_loc[model_i]  # 예측 위치 저장
            pos_values[model_i][steps] = position[model_i]

        # 객체 업데이트 및 그리기
        for obj in objects:
            obj.update(0.1)
            obj.draw(screen)
            draw_offset(screen, obj)   # obj 1, 2, 3에 대한 offset 표현
            draw_velocity_arrows(screen, obj, center_x, center_y)   # obj 1, 2, 3에 대한 속도 화살표 표현


        pygame.display.flip()

    IMM.draw_pos(time_steps, pos_values, predicted_loc_values, num_objects)
    imm[0].draw_model_prob(1, "OB", 0)
    imm[1].draw_model_prob(1, "OB", 1)
    imm[2].draw_model_prob(1, "OB", 2)

    ######### 위협도 출력 ##########
    Threat.draw_threat(time_steps, threat_values, num_objects)
    ##############################

    pygame.quit()


def main_timesteps_simple(time_steps):
    """
    좀더 모듈화 된 메인 코드
    :param time_steps: 실행할 스탭 수
    :return:
    """
    init_settings()
    screen, clock = init_pygame()

    # 객체 생성 (ID 부여)
    num_objects = 3  # 다중 객체 수
    num_ROI = 3 #구역 수
    objects = [DynamicObject(id=i + 1) for i in range(num_objects)]   # 객체 마릿수 지정

    threat_values = np.zeros((num_objects, time_steps))

    ### IMM 초기화 부분 ###
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


    #################### IMM 실행 ###########################
    for steps in range(time_steps):
        screen.fill(WHITE)
        # 동심원 그리기
        draw_circles(screen)
        # 중심점 설정
        center_x, center_y = WIDTH // 2, HEIGHT // 2

        ### 각 스텝마다 개체별로 위치, 속도 저장 ###
        for i in range(num_objects):
            position[i] = objects[i].r
            pos_values[i][steps] = position[i]
        for obj in range(num_objects):
            objects[obj].draw(screen)
            velocity_values[obj] = objects[obj].radial_speed

        ### 위협 평가 ###
        for obj in range(num_objects):
            if position[obj]<=2:
                position[obj] = 2.001
            threat_values[obj][steps] = Threat.cal_threat(12, position[obj], velocity_values[obj])
        ###############

        print("=============================================================")
        print("{}번째 IMM 진행".format(steps+1))

        predicted_loc, predicted_cov, filtered_mu = IMM.predict_onestep(num_objects, model_probs, position, velocity_values)
        model_probs = filtered_mu

        for i in range(num_objects):
            predicted_loc_values[i][steps+1] = predicted_loc[i]
            predicted_cov_values[i][steps+1] = predicted_cov[i]

        # 객체 업데이트 및 그리기
        for obj in objects:
            obj.update(0.1)
            obj.draw(screen)
            draw_offset(screen, obj)   # obj 1, 2, 3에 대한 offset 표현
            draw_velocity_arrows(screen, obj, center_x, center_y)   # obj 1, 2, 3에 대한 속도 화살표 표현


        pygame.display.flip()

    IMM.draw_pos(time_steps, pos_values, predicted_loc_values, num_objects)
    IMM.cov_print(time_steps, num_objects, predicted_cov_values)

    ######### 위협도 출력 ##########
    Threat.draw_threat(time_steps, threat_values, num_objects)
    ##############################

    pygame.quit()



# 실행
if __name__ == "__main__":
    #main_timesteps(100)
    main_timesteps_simple(100)
