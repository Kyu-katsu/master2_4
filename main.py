import time
import pygame
import numpy as np
import math


# 초기 설정
def init_settings():
    global WIDTH, HEIGHT, SPACE_SIZE, SCALE, WHITE, RED, GREEN, BLUE
    WIDTH, HEIGHT = 800, 800
    SPACE_SIZE = 100
    SCALE = WIDTH // SPACE_SIZE

    # 색상 정의
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)


# Pygame 초기화 및 화면 설정
def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    return screen, clock


# 동적 객체 클래스 정의
class DynamicObject:
    def __init__(self):
        self.x = np.random.uniform(0, SPACE_SIZE)
        self.y = np.random.uniform(0, SPACE_SIZE)
        self.speed = np.random.uniform(0, 5)
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.ax = np.random.uniform(-0.5, 0.5)
        self.ay = np.random.uniform(-0.5, 0.5)

    def update(self, dt):
        # 가속도 업데이트
        self.ax += np.random.uniform(-0.2, 0.2)
        self.ay += np.random.uniform(-0.2, 0.2)
        
        # 가속도 업데이트 된건 맞는 것 같은데, 좀 잘 생각해봐야될듯. 환경을 어떻게 구성해야 되는건지.
        # 일단 객체가 사람처럼 이동하려면...? 가속도가 영향이 좀 있어야 하는거 아닌가?
        # 속도를 아무렇게나 설정하는게 맞나? 가속도를 아무렇게나 설정하는게 맞나? 가속도도 업데이트를 진행하면 안되는 건가?
        # 방향만 바꾸는건 말이 안됨. 속력도 바뀌어야됨
        
        # 가속도 적용
        # self.speed += np.sqrt(self.ax ** 2 + self.ay ** 2) * dt   # 가속도 미적용
        # self.speed += (self.ax * np.cos(self.direction) + self.ay * np.sin(self.direction)) * dt
        # self.direction += np.random.normal(0, 0.1)  # 방향에 약간의 랜덤성 추가

        # 속도 업데이트
        self.speed_x = self.speed * np.cos(self.direction) + self.ax * dt
        self.speed_y = self.speed * np.sin(self.direction) + self.ay * dt

        # 속도: 방향과 크기 계산
        self.direction = math.atan2(self.speed_y, self.speed_x)
        self.speed = np.sqrt(self.speed_x**2 + self.speed_y**2)

        # 속도 제한
        self.speed = np.clip(self.speed, 0, 5)

        # 위치 업데이트
        self.x += self.speed * np.cos(self.direction) * dt
        self.y += self.speed * np.sin(self.direction) * dt

        # 경계 처리
        if self.x < 0 or self.x > SPACE_SIZE:
            self.direction = np.pi - self.direction
        if self.y < 0 or self.y > SPACE_SIZE:
            self.direction = -self.direction

        self.x = np.clip(self.x, 0, SPACE_SIZE)
        self.y = np.clip(self.y, 0, SPACE_SIZE)

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x * SCALE), int(self.y * SCALE)), 5)

    def calculate_offset(self):
        center_x, center_y = SPACE_SIZE // 2, SPACE_SIZE // 2
        return math.sqrt((self.x - center_x) ** 2 + (self.y - center_y) ** 2)

    @property
    def vx(self):
        return self.speed * np.cos(self.direction)

    @property
    def vy(self):
        return self.speed * np.sin(self.direction)


# 동심원 그리기 함수
def draw_circles(screen):
    pygame.draw.circle(screen, BLUE, (WIDTH // 2, HEIGHT // 2), 50 * SCALE, 1)
    pygame.draw.circle(screen, GREEN, (WIDTH // 2, HEIGHT // 2), 30 * SCALE, 1)
    pygame.draw.circle(screen, RED, (WIDTH // 2, HEIGHT // 2), 10 * SCALE, 1)


# 오프셋 계산 및 표시 함수
def draw_offset(screen, obj):
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    offset = math.sqrt((obj.x * SCALE - center_x) ** 2 + (obj.y * SCALE - center_y) ** 2)
    font = pygame.font.Font(None, 24)
    text = font.render(f"Offset: {offset / SCALE:.2f}", True, (0, 0, 0))
    screen.blit(text, (obj.x * SCALE, obj.y * SCALE - 20))


def log_object_states(objects):
    print("\nObject States:")
    for i, obj in enumerate(objects):
        offset = obj.calculate_offset()
        print(f"Object {i+1}: x={obj.x:.2f}, y={obj.y:.2f}, "
              f"vx={obj.vx:.2f}, vy={obj.vy:.2f}, "
              f"ax={obj.ax:.2f}, ay={obj.ay:.2f}, offset={offset:.2f}")


# 메인 루프
def main():
    init_settings()
    screen, clock = init_pygame()

    # 객체 생성
    objects = [DynamicObject() for _ in range(3)]
    last_log_time = time.time()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        # 동심원 그리기
        draw_circles(screen)

        # 객체 업데이트 및 그리기
        for obj in objects:
            obj.update(0.1)
            obj.draw(screen)
            draw_offset(screen, obj)

        current_time = time.time()
        if current_time - last_log_time >= 1.0:
            log_object_states(objects)
            last_log_time = current_time

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# 실행
if __name__ == "__main__":
    main()
