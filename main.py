import pygame
import numpy as np
import math

# 초기 설정
WIDTH, HEIGHT = 800, 800
SPACE_SIZE = 100
SCALE = WIDTH // SPACE_SIZE

# 색상 정의
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

class DynamicObject:
    def __init__(self):
        self.x = np.random.uniform(0, SPACE_SIZE)
        self.y = np.random.uniform(0, SPACE_SIZE)
        self.speed = np.random.uniform(0, 5)
        self.direction = np.random.uniform(0, 2*np.pi)
        self.ax = 0
        self.ay = 0

    def update(self, dt):
        # 가속도 적용
        self.speed += np.sqrt(self.ax**2 + self.ay**2) * dt
        self.direction += np.random.normal(0, 0.1)  # 방향에 약간의 랜덤성 추가

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

    @property
    def vx(self):
        return self.speed * np.cos(self.direction)

    @property
    def vy(self):
        return self.speed * np.sin(self.direction)

print('git 연결 확인됐나?')

# 객체 생성
objects = [DynamicObject() for _ in range(3)]

# 메인 루프
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # 동심원 그리기
    pygame.draw.circle(screen, BLUE, (WIDTH//2, HEIGHT//2), 50 * SCALE, 1)
    pygame.draw.circle(screen, GREEN, (WIDTH//2, HEIGHT//2), 30 * SCALE, 1)
    pygame.draw.circle(screen, RED, (WIDTH//2, HEIGHT//2), 10 * SCALE, 1)

    # 객체 업데이트 및 그리기
    for obj in objects:
        obj.update(0.1)
        obj.draw(screen)

        # 횡방향 오프셋 계산 및 표시
        center_x, center_y = WIDTH//2, HEIGHT//2
        offset = math.sqrt((obj.x * SCALE - center_x)**2 + (obj.y * SCALE - center_y)**2)
        font = pygame.font.Font(None, 24)
        text = font.render(f"Offset: {offset/SCALE:.2f}", True, (0, 0, 0))
        screen.blit(text, (obj.x * SCALE, obj.y * SCALE - 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()