import pygame
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'

# 초기 설정 함수
def init_settings():
    global WIDTH, HEIGHT, SPACE_SIZE, SCALE
    global center_x_pix, center_y_pix, actual_center_x, actual_center_y
    # 화면 크기 (픽셀 단위)
    WIDTH, HEIGHT = 800, 800
    # 실제 공간 크기 (m 단위); 실제 좌표 범위는 x, y ∈ [–12, 12] (총 24m)
    SPACE_SIZE = 24
    # 1m 당 픽셀 수: 800 픽셀 / 24 m
    SCALE = WIDTH / SPACE_SIZE  # 약 33.33 픽셀/m

    # 픽셀 좌표계에서 실제 중심 (0,0)에 해당하는 픽셀 좌표
    center_x_pix = WIDTH // 2
    center_y_pix = HEIGHT // 2

    # 실제 공간의 중심은 (0, 0)
    actual_center_x, actual_center_y = 0, 0

    # 색상 정의 (RGB 튜플)
    global WHITE, RED, ORANGE, YELLOW, GREEN, BLUE, BLACK
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)


# Pygame 초기화 함수: 화면과 시계(clock)를 생성
def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    return screen, clock


# 실제 공간 좌표(미터)를 픽셀 좌표로 변환하는 함수
def actual_to_pixel(x, y):
    """
    실제 공간 좌표 (x, y) [m 단위]를 픽셀 좌표로 변환합니다.
    x_pixel = center_x_pix + (x * SCALE)
    y_pixel = center_y_pix - (y * SCALE)   (y는 위쪽이 양수)
    """
    # center_x_pix, center_y_pix = 400, 400
    # scale = 33.33
    pixel_x = int(center_x_pix + x * SCALE)
    pixel_y = int(center_y_pix - y * SCALE)     # y가 위로 갈수록 양수여서 부호 반대로 설정.
    return pixel_x, pixel_y


# 동적 객체 클래스: 실제 좌표를 기반으로 운동을 계산
class DynamicObject:
    def __init__(self, id):
        self.id = id  # 객체 ID (예: 0, 1, 2 등)

        # 극좌표계를 이용하여 초기 실제 위치 (미터 단위)를 생성
        # r는 2m ~ 10m 사이, theta는 0 ~ 2π 사이에서 랜덤 선택
        self.r = np.random.uniform(2, 10)
        self.theta = np.random.uniform(0, 2 * np.pi)
        # 실제 좌표: (x, y) = (r cosθ, r sinθ)
        self.actual_x = self.r * np.cos(self.theta)
        self.actual_y = self.r * np.sin(self.theta)

        # 속력 (m/s), 운동 방향 (라디안)
        self.speed = np.random.uniform(0, 1)   # m/s 단위
        self.direction = np.random.uniform(0, 2 * np.pi)
        # 가속도 (m/s²)
        self.ax = np.random.uniform(-0.5, 0.5)
        self.ay = np.random.uniform(-0.5, 0.5)

        # Return 변수
        self.radial_speed = 0
        self.offset = 0

    def update(self, dt):
        """
        dt: 시간 간격 (초)
        실제 좌표에서의 운동(가속도, 속도, 위치 업데이트)을 수행하고,
        만약 객체의 중심에서의 거리가 2m 미만 또는 10m 초과라면 속도 벡터를 반사하여 경계 내로 유지합니다.
        """
        # 1. 가속도 업데이트: 약간의 랜덤 변화 후 -0.5 ~ 0.5 범위로 제한
        self.ax = np.clip(self.ax + np.random.uniform(-0.05, 0.05), -0.5, 0.5)
        self.ay = np.clip(self.ay + np.random.uniform(-0.05, 0.05), -0.5, 0.5)

        # 2. 속도 성분 계산 (m/s 단위)
        #    기존 속도에 가속도 성분을 더하여 새 속도 성분 계산
        speed_x = self.speed * np.cos(self.direction) + self.ax * dt
        speed_y = self.speed * np.sin(self.direction) + self.ay * dt

        # 3. 새 방향 계산: 현재 속도 성분의 각도를 이용하여
        target_direction = math.atan2(speed_y, speed_x)
        # 현재 방향과 목표 방향을 혼합하여 점진적으로 변경하고, 작은 랜덤 변화 추가
        self.direction = 0.9 * self.direction + 0.1 * target_direction + np.random.normal(0, 0.05)

        # 4. 새 속력 계산 (magnitude) 및 속력 제한 (0 ~ 1 m/s)
        self.speed = np.sqrt(speed_x ** 2 + speed_y ** 2)
        self.speed = np.clip(self.speed, 0, 1)

        # 5. 실제 좌표 업데이트: 실제 좌표 (미터 단위)에서 위치 이동
        self.actual_x += self.speed * np.cos(self.direction) * dt
        self.actual_y += self.speed * np.sin(self.direction) * dt

        # 6. 경계 검사: 객체의 현재 거리 계산 (중심에서의 거리)
        current_r = math.sqrt(self.actual_x ** 2 + self.actual_y ** 2)

        # 7. 만약 객체가 영역 밖이면 (2m 미만 또는 10m 초과)
        if current_r < 2 or current_r > 10:
            # 현재 속도 벡터 계산
            vx = self.speed * np.cos(self.direction)
            vy = self.speed * np.sin(self.direction)
            # 중심에서 객체까지의 위치 벡터의 단위 벡터 계산 (n)
            n_x = self.actual_x / current_r
            n_y = self.actual_y / current_r
            # 속도 벡터 반사: v' = v - 2*(v·n)*n
            dot = vx * n_x + vy * n_y  # v와 n의 내적
            new_vx = vx - 2 * dot * n_x
            new_vy = vy - 2 * dot * n_y
            # 반사된 속도 벡터를 이용해 새로운 방향을 설정
            self.direction = math.atan2(new_vy, new_vx)
            # 필요에 따라, 위치가 경계를 벗어난 경우 경계에 맞게 재설정:
            if current_r < 2:
                # 내측 경계: 실제 위치를 2m에 고정
                self.actual_x = 2 * n_x
                self.actual_y = 2 * n_y
            elif current_r > 10:
                # 외측 경계: 실제 위치를 10m에 고정
                self.actual_x = 10 * n_x
                self.actual_y = 10 * n_y

        self.offset = math.sqrt(self.actual_x ** 2 + self.actual_y ** 2)


    def draw(self, screen):
        """
        실제 좌표를 픽셀 좌표로 변환하여 객체를 그립니다.
        객체는 빨간 원으로 표시되며, ID 번호도 함께 출력합니다.
        """
        # 실제 좌표 -> 픽셀 좌표 변환
        pixel_pos = actual_to_pixel(self.actual_x, self.actual_y)
        # 객체 원 그리기 (반지름 5 픽셀)
        pygame.draw.circle(screen, RED, pixel_pos, 5)

        # 객체 번호 표시
        font = pygame.font.Font(None, 24)
        text = font.render(str(self.id), True, BLACK)
        # ID 텍스트를 객체 오른쪽 위에 출력
        screen.blit(text, (pixel_pos[0] + 10, pixel_pos[1] - 10))

    @property
    def vx(self):
        # x축 속도 (m/s)
        return self.speed * np.cos(self.direction)

    @property
    def vy(self):
        # y축 속도 (m/s)
        return self.speed * np.sin(self.direction)


# 동심원(구역) 그리기 함수: 실제 공간의 동심원 구역을 픽셀로 변환하여 그림
def draw_circles(screen):
    """
    24m×24m 실제 공간에서 중심(0,0)을 기준으로
      - 반지름 12m: 전체 영역 (노란색)
      - 반지름 8m : 중간 영역 (주황색)
      - 반지름 4m : 내부 영역 (빨간색)
    등을 색으로 채워서 표현합니다.
    픽셀 좌표로 변환하여 화면에 출력합니다.
    """
    # 실제 반지름을 m 단위로 설정하고, 픽셀로 변환:
    # 픽셀 반지름 = 실제 반지름 * SCALE
    # 중심은 center_x_pix, center_y_pix
    pygame.draw.circle(screen, YELLOW, (center_x_pix, center_y_pix), int(12 * SCALE))
    pygame.draw.circle(screen, ORANGE, (center_x_pix, center_y_pix), int(8 * SCALE))
    pygame.draw.circle(screen, RED, (center_x_pix, center_y_pix), int(4 * SCALE))
    # 외곽선 그리기 (검은색 선)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(12 * SCALE), 3)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(10 * SCALE), 1)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(8 * SCALE), 3)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(6 * SCALE), 1)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(4 * SCALE), 3)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(2 * SCALE), 1)


# 오프셋(중심과의 거리) 계산 및 화면에 표시하는 함수
def draw_offset(screen, obj):
    """
    객체와 실제 공간 중심(0,0) 간의 거리를 계산하고,
    이를 실제 m 단위로 변환하여 화면에 표시합니다.
    """
    # 실제 중심은 (0,0) -> 픽셀 중심은 (center_x_pix, center_y_pix)
    pixel_pos = actual_to_pixel(obj.actual_x, obj.actual_y)
    offset_pixels = math.sqrt((pixel_pos[0] - center_x_pix)**2 + (pixel_pos[1] - center_y_pix)**2)
    offset_m = offset_pixels / SCALE  # 픽셀 단위를 m 단위로 변환
    font = pygame.font.Font(None, 24)
    text = font.render(f"Offset: {offset_m:.2f} m", True, BLACK)
    screen.blit(text, (pixel_pos[0], pixel_pos[1] - 20))


# 속도 화살표 그리기 함수: 객체의 속도를 x축, y축 및 중심 방향으로 시각화
def draw_velocity_arrows(screen, obj):
    """
    객체의 현재 위치에서,
      - 녹색: x축 속도
      - 파란색: y축 속도
      - 빨간색: 중심 방향 속도 (중심과 객체 간의 벡터에 따른 성분)
    을 화살표로 그립니다.
    """
    # 실제 좌표 -> 픽셀 좌표 변환
    start_pos = actual_to_pixel(obj.actual_x, obj.actual_y)

    # 내부 함수: 주어진 벡터를 화살표로 그림
    def draw_arrow(color, start, vector, scale=100):
        magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)  # 벡터 크기
        if magnitude == 0:
            return  # 속도가 0이면 화살표를 그리지 않음
        unit_vector = (vector[0] / magnitude, vector[1] / magnitude)
        end_pos = (start[0] + int(unit_vector[0] * magnitude * scale),
                   start[1] + int(unit_vector[1] * magnitude * scale))
        pygame.draw.line(screen, color, start, end_pos, 2)
        # 화살표 촉 그리기
        arrow_size = 10
        arrow_point1 = (end_pos[0] - int(unit_vector[0] * arrow_size - unit_vector[1] * arrow_size // 2),
                        end_pos[1] - int(unit_vector[1] * arrow_size + unit_vector[0] * arrow_size // 2))
        arrow_point2 = (end_pos[0] - int(unit_vector[0] * arrow_size + unit_vector[1] * arrow_size // 2),
                        end_pos[1] - int(unit_vector[1] * arrow_size - unit_vector[0] * arrow_size // 2))
        pygame.draw.polygon(screen, color, [end_pos, arrow_point1, arrow_point2])

    # x축 속도 화살표 (녹색)
    draw_arrow(GREEN, start_pos, (obj.vx, 0))
    # y축 속도 화살표 (파란색)
    draw_arrow(BLUE, start_pos, (0, obj.vy))
    # 중심점 방향 속력 화살표 (검은색)
    draw_arrow(BLACK, start_pos, (obj.vx, obj.vy))

    # # 중심 방향 속도 화살표 (빨간색)
    # # 중심 (실제 좌표 (0,0)) -> 픽셀 중심은 (center_x_pix, center_y_pix)
    # # 실제 중심과 객체 간의 벡터 (실제 m 단위)
    # vector_to_center = (actual_center_x - obj.actual_x, actual_center_y - obj.actual_y)
    # mag_center = math.sqrt(vector_to_center[0]**2 + vector_to_center[1]**2)
    # if mag_center > 0:
    #     unit_center = (vector_to_center[0] / mag_center, vector_to_center[1] / mag_center)
    # else:
    #     unit_center = (0, 0)
    # # 객체 속도의 중심 방향 성분: 내적
    # center_speed = obj.vx * unit_center[0] + obj.vy * unit_center[1]
    # center_vector = (unit_center[0] * center_speed, unit_center[1] * center_speed)
    # draw_arrow(RED, start_pos, center_vector)
    # 중심 방향 속도 화살표 (빨간색)
    # 중심으로 향하는 속력을 계산하는 함수 사용

    center_speed = compute_center_velocity(obj)
    # 중심 방향 단위벡터: (0,0)에서 객체 위치를 향하는 벡터의 반대(즉, 실제 중심을 향함)
    mag = math.sqrt(obj.actual_x**2 + obj.actual_y**2)
    if mag > 0:
        unit_center = (-obj.actual_x / mag, -obj.actual_y / mag)
    else:
        unit_center = (0, 0)
    # 중심 방향 성분 벡터 (m/s)
    center_vector = (unit_center[0] * center_speed, unit_center[1] * center_speed)
    draw_arrow(RED, start_pos, center_vector)


def compute_center_velocity(obj):
    """
    객체 obj의 실제 속도 벡터(obj.vx, obj.vy)와,
    실제 중심 (actual_center_x, actual_center_y)과 객체 위치 (obj.actual_x, obj.actual_y)로부터
    중심으로 향하는 단위 벡터를 구합니다.

    그 후, 객체 속도의 중심 방향 성분을 내적으로 계산하여,
    만약 음수(즉, 중심에서 멀어지는 방향)라면 0을 반환합니다.

    반환 값은 중심으로 향하는 속력 (m/s)입니다.
    """
    # 중심은 (0,0)인 경우:
    vector_to_center = (-obj.actual_x, -obj.actual_y)
    mag = math.sqrt(vector_to_center[0] ** 2 + vector_to_center[1] ** 2)
    if mag == 0:
        return 0
    unit_vector = (vector_to_center[0] / mag, vector_to_center[1] / mag)
    # 속도 성분: 내적 (객체가 중심으로 움직이면 양수, 멀어지면 음수)
    center_speed = obj.vx * unit_vector[0] + obj.vy * unit_vector[1]
    # 만약 음수라면, 중심으로 향하는 속력은 0으로 처리
    return max(0, center_speed)


# 메인 함수: 설정 초기화, 객체 생성, 메인 루프 실행
def main():
    init_settings()
    screen, clock = init_pygame()

    # 객체 생성: 실제 공간 좌표를 기반으로 초기화된 동적 객체들 (ID는 0,1,2로 할당)
    num_objects = 3
    objects = [DynamicObject(id=i) for i in range(num_objects)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        # 동심원(구역) 그리기: 실제 공간을 기준으로 색칠된 구역을 화면에 표시
        draw_circles(screen)

        # 각 객체에 대해: 화면에 그리기, offset 및 속도 화살표 표현, 그리고 실제 좌표 업데이트
        for obj in objects:
            # 디버그 출력 (콘솔에 실제 좌표와 속도 출력)
            print(f"Object ID: {obj.id}, lateral offset: ({obj.offset:.2f}) Actual Pos: ({obj.actual_x:.2f}, {obj.actual_y:.2f}), Speed: {obj.speed:.2f}")
            obj.draw(screen)
            draw_offset(screen, obj)
            draw_velocity_arrows(screen, obj)
            obj.update(0.1)

        pygame.display.flip()
        clock.tick(60)  # 초당 60 프레임

    pygame.quit()


# 프로그램 실행
if __name__ == "__main__":
    main()
