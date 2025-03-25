import pygame
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.stats import norm


# 초기 설정
def init_settings():
    global WIDTH, HEIGHT, SPACE_SIZE, SCALE, center_x_pix, center_y_pix
    WIDTH, HEIGHT = 800, 800
    SPACE_SIZE = 24
    SCALE = WIDTH // SPACE_SIZE
    center_x_pix, center_y_pix = WIDTH // 2, HEIGHT // 2

    # 색상 정의
    global WHITE, RED, ORANGE, YELLOW, GREEN, BLUE, BLACK, PURPLE
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    PURPLE = (128, 0, 128)


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

        # 극좌표계를 이용하여 초기 실제 위치 (미터 단위)를 생성
        # r는 2m ~ 10m 사이, theta는 0 ~ 2π 사이에서 랜덤 선택
        self.r = np.random.uniform(2, 10)
        self.theta = np.random.uniform(0, 2 * np.pi)
        # 실제 좌표: (x, y) = (r cosθ, r sinθ)

        self.x = self.r * np.cos(self.theta)
        self.y = self.r * np.sin(self.theta)
        self.speed = np.random.uniform(0, 1)
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.ax = np.random.uniform(-0.5, 0.5)
        self.ay = np.random.uniform(-0.5, 0.5)

        # Return 변수
        self.offset = math.sqrt(self.x ** 2 + self.y ** 2)  ### q_k
        # 속도 벡터 (x, y 성분)
        vx = self.speed * np.cos(self.direction)
        vy = self.speed * np.sin(self.direction)
        # 중심방향 벡터의 단위벡터 (n_x, n_y)
        n_x = self.x / self.offset  # 현재 위치를 중심으로 정규화
        n_y = self.y / self.offset
        self.actual_center_vel = vx * n_x + vy * n_y    ### dot_q_k

    def update(self, dt):
        # 1. 가속도 업데이트
        self.ax = np.clip(self.ax + np.random.uniform(-0.05, 0.05), -0.5, 0.5)
        self.ay = np.clip(self.ay + np.random.uniform(-0.05, 0.05), -0.5, 0.5)

        # 2. 속도 성분 계산 (m/s)
        speed_x = self.speed * np.cos(self.direction) + self.ax * dt
        speed_y = self.speed * np.sin(self.direction) + self.ay * dt

        # 3. 새 방향 계산
        target_direction = math.atan2(speed_y, speed_x)
        self.direction = 0.9 * self.direction + 0.1 * target_direction + np.random.normal(0, 0.05)

        # 4. 새 속력 계산 및 제한
        self.speed = np.sqrt(speed_x ** 2 + speed_y ** 2)
        self.speed = np.clip(self.speed, 0, 1)

        # 5. 위치 업데이트 (실제 좌표, m 단위)
        self.x += self.speed * np.cos(self.direction) * dt
        self.y += self.speed * np.sin(self.direction) * dt

        # 6. 경계 검사: 중심에서의 현재 거리 계산
        current_r = math.sqrt(self.x ** 2 + self.y ** 2)
        if current_r < 2 or current_r > 10:
            vx = self.speed * np.cos(self.direction)
            vy = self.speed * np.sin(self.direction)
            n_x = self.x / current_r
            n_y = self.y / current_r
            dot = vx * n_x + vy * n_y
            new_vx = vx - 2 * dot * n_x
            new_vy = vy - 2 * dot * n_y
            self.direction = math.atan2(new_vy, new_vx)
            if current_r < 2:
                self.x = 2 * n_x
                self.y = 2 * n_y
            elif current_r > 10:
                self.x = 10 * n_x
                self.y = 10 * n_y

        # 7. offset 업데이트 및 중심 방향 속력 재계산
        self.offset = math.sqrt(self.x ** 2 + self.y ** 2)
        vx = self.speed * np.cos(self.direction)
        vy = self.speed * np.sin(self.direction)
        # unit vector from object to center = (-x/offset, -y/offset)
        n_x = -self.x / self.offset
        n_y = -self.y / self.offset
        self.actual_center_vel = vx * n_x + vy * n_y

    def draw(self, screen):
        # 객체 원 그리기
        pygame.draw.circle(screen, BLACK, (int(center_x_pix + self.x * SCALE), int(center_y_pix - self.y * SCALE)), 5)

        # 객체 번호 표시
        font = pygame.font.Font(None, 24)
        text = font.render(str(self.id), True, (0, 0, 0))
        screen.blit(text, (int(center_x_pix + self.x * SCALE) + 10, int(center_y_pix - self.y * SCALE) - 10))


    @property
    def vx(self):
        return self.speed * np.cos(self.direction)

    @property
    def vy(self):
        return self.speed * np.sin(self.direction)


# 동심원 그리기 함수
def draw_circles(screen):
    pygame.draw.circle(screen, YELLOW, (center_x_pix, center_y_pix), int(12 * SCALE))
    pygame.draw.circle(screen, ORANGE, (center_x_pix, center_y_pix), int(8 * SCALE))
    pygame.draw.circle(screen, RED, (center_x_pix, center_y_pix), int(4 * SCALE))
    # 선 그리기 (검은색 선)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(12 * SCALE), 3)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(10 * SCALE), 1)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(8 * SCALE), 3)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(6 * SCALE), 1)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(4 * SCALE), 3)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(2 * SCALE), 1)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(0.1 * SCALE), 3)


# 오프셋 계산 및 표시 함수
def draw_offset(screen, obj):
    font = pygame.font.Font(None, 24)
    text = font.render(f"Offset: {obj.offset:.2f}", True, BLACK)
    screen.blit(text, (int(center_x_pix + obj.x * SCALE), int(center_y_pix - obj.y * SCALE) - 20))


# 속도 화살표 그림 함수
def draw_velocity_arrows(screen, obj):
    # 인라인으로 픽셀 좌표 계산 (실제 좌표 → 픽셀)
    start_pos = (int(center_x_pix + obj.x * SCALE), int(center_y_pix - obj.y * SCALE))

    def draw_arrow(color, start, vector, scale=80):
        magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
        if magnitude == 0:
            return
        unit_vector = (vector[0] / magnitude, vector[1] / magnitude)
        end_pos = (start[0] + int(unit_vector[0] * magnitude * scale),
                   start[1] + int(unit_vector[1] * magnitude * scale))
        pygame.draw.line(screen, color, start, end_pos, 2)
        arrow_size = 10
        arrow_point1 = (end_pos[0] - int(unit_vector[0] * arrow_size - unit_vector[1] * arrow_size // 2),
                        end_pos[1] - int(unit_vector[1] * arrow_size + unit_vector[0] * arrow_size // 2))
        arrow_point2 = (end_pos[0] - int(unit_vector[0] * arrow_size + unit_vector[1] * arrow_size // 2),
                        end_pos[1] - int(unit_vector[1] * arrow_size - unit_vector[0] * arrow_size // 2))
        pygame.draw.polygon(screen, color, [end_pos, arrow_point1, arrow_point2])

    # 녹색: x축 속도 성분 (실제 속도, m/s → 픽셀 단위; x는 그대로)
    draw_arrow(GREEN, start_pos, (obj.vx, 0))
    # 파란색: y축 속도 성분 (픽셀 좌표계에서 y 반전)
    draw_arrow(BLUE, start_pos, (0, -obj.vy))
    # 검은색: 전체 속도 벡터 (실제 속도, y축 부호 반전)
    draw_arrow(BLACK, start_pos, (obj.vx, -obj.vy))

    # 빨간색: 중심 방향 속도 벡터
    # 중심으로 향하는 단위벡터: from object to center = (-x/offset, -y/offset)
    r = math.sqrt(obj.x**2 + obj.y**2)
    if r > 0:
        unit_center = (-obj.x / r, -obj.y / r)
    else:
        unit_center = (0, 0)
    # 실제 중심 방향 속력: 내적 (양이면 객체가 중심으로, 음이면 반대 방향)
    proj = obj.vx * unit_center[0] + obj.vy * unit_center[1]
    # 변환: 픽셀 단위로 변환 시, y축 부호 반전
    center_vector = (unit_center[0] * proj, -unit_center[1] * proj)
    draw_arrow(PURPLE, start_pos, center_vector)

# 한 타임스텝씩 시뮬레이션하는 함수
def simulate_onestep(objects, dt, screen):
    """
    objects: DynamicObject 리스트
    dt: 시간 간격 (초)
    screen: Pygame 화면
    각 객체에 대해 업데이트, 그리기, offset 및 중심 방향 속력 표시.
    반환: (offsets, center_vels) 각각 리스트 (순서: 객체 id 순)
    """
    for obj in objects:
        obj.update(dt)
    # 화면 그리기
    screen.fill(WHITE)
    draw_circles(screen)
    offsets = []
    center_vels = []
    for obj in objects:
        obj.draw(screen)
        draw_offset(screen, obj)
        draw_velocity_arrows(screen, obj)
        offsets.append(obj.offset)
        center_vels.append(obj.actual_center_vel)
    pygame.display.flip()
    return offsets, center_vels



def simulate_steps(num_steps, dt=0.1, num_objects=3):
    """
    num_steps: 시뮬레이션 할 타임 스텝 수
    dt: 각 스텝의 시간 간격 (초)
    num_objects: 생성할 객체 수
    반환:
       offsets: shape (num_objects, num_steps), 각 객체의 offset (m)
       center_vels: shape (num_objects, num_steps), 각 객체의 중심 방향 속력 (m/s)
    """
    # 초기 설정
    init_settings()
    # 객체 생성
    objects = [DynamicObject(id=i) for i in range(num_objects)]
    # 결과 저장 배열 생성
    offsets = np.zeros((num_objects, num_steps))
    center_vels = np.zeros((num_objects, num_steps))
    # Pygame 창 생성 (옵션: 시각화 없이 시뮬레이션만 진행 가능)
    screen, clock = init_pygame()
    # 시뮬레이션 루프: 정해진 스텝 수만큼 업데이트 후 결과 기록
    for t in range(num_steps):
        # 화면 업데이트 (필요하면 주석 처리 가능)
        screen.fill(WHITE)
        draw_circles(screen)
        for obj in objects:
            obj.update(dt)
            offsets[obj.id, t] = obj.offset
            center_vels[obj.id, t] = obj.actual_center_vel
            # 시각화를 원하면 객체 그리기
            obj.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    return offsets, center_vels


# def main():
#     init_settings()
#     screen, clock = init_pygame()
#
#     # 객체 생성 (ID 부여)
#     num_objects = 3  # 다중 객체 수
#     objects = [DynamicObject(id=i + 1) for i in range(num_objects)]   # 객체 마릿수 지정
#
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#
#         screen.fill(WHITE)
#
#         # 동심원 그리기
#         draw_circles(screen)
#
#         for obj in objects:
#             obj.update(0.1)
#             print(f"Object ID: {obj.id}, lateral offset: ({obj.offset:.2f}) Actual Pos: ({obj.x:.2f}, {obj.y:.2f}), Speed: {obj.speed:.2f}, Center vel: {obj.actual_center_vel}")
#             obj.draw(screen)
#             draw_offset(screen, obj)   # obj 1, 2, 3에 대한 offset 표현
#             draw_velocity_arrows(screen, obj)   # obj 1, 2, 3에 대한 속도 화살표 표현
#
#         pygame.display.flip()  # 화면 업데이트
#         clock.tick(60)  # 초당 30프레임 유지
#
#     pygame.quit()
#
# # 실행
# if __name__ == "__main__":
#     main()
