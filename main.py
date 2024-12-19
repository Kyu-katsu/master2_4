import time
import pygame
import numpy as np
import math


# 초기 설정
def init_settings():
    global WIDTH, HEIGHT, SPACE_SIZE, SCALE, WHITE, RED, GREEN, BLUE, BLACK
    WIDTH, HEIGHT = 800, 800
    SPACE_SIZE = 100
    SCALE = WIDTH // SPACE_SIZE

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
        self.x = np.random.uniform(0, SPACE_SIZE)
        self.y = np.random.uniform(0, SPACE_SIZE)
        self.speed = np.random.uniform(0, 5)
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.ax = np.random.uniform(-0.5, 0.5)
        self.ay = np.random.uniform(-0.5, 0.5)

    def update(self, dt):
        # 가속도 업데이트
        self.ax = np.clip(self.ax + np.random.uniform(-0.05, 0.05), -0.5, 0.5)
        self.ay = np.clip(self.ay + np.random.uniform(-0.05, 0.05), -0.5, 0.5)

        # 속도 업데이트
        self.speed_x = self.speed * np.cos(self.direction) + self.ax * dt
        self.speed_y = self.speed * np.sin(self.direction) + self.ay * dt

        # 방향 계산: 기존 방향 + 랜덤 변동
        target_direction = math.atan2(self.speed_y, self.speed_x)
        self.direction = 0.9 * self.direction + 0.1 * target_direction  # 점진적 변경
        self.direction += np.random.normal(0, 0.05)  # 작은 랜덤 변화 추가

        # 속력 계산: 제한 범위 설정
        self.speed = np.sqrt(self.speed_x**2 + self.speed_y**2)
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
        # 객체 원 그리기
        pygame.draw.circle(screen, RED, (int(self.x * SCALE), int(self.y * SCALE)), 5)

        # 객체 번호 표시
        font = pygame.font.Font(None, 24)
        text = font.render(str(self.id), True, (0, 0, 0))
        screen.blit(text, (int(self.x * SCALE) + 10, int(self.y * SCALE) - 10))

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
    pygame.draw.circle(screen, BLUE, (WIDTH // 2, HEIGHT // 2), 50 * SCALE, 1)   # RoI 3
    pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 40 * SCALE, 1)  # RoI 3 중심선
    pygame.draw.circle(screen, GREEN, (WIDTH // 2, HEIGHT // 2), 30 * SCALE, 1)  # RoI 2
    pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 20 * SCALE, 1)  # RoI 2 중심선
    pygame.draw.circle(screen, RED, (WIDTH // 2, HEIGHT // 2), 10 * SCALE, 1)    # RoI 1
    pygame.draw.circle(screen, BLACK, (WIDTH // 2, HEIGHT // 2), 5 * SCALE, 1)   # RoI 1 중심선


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


def log_offsets(timestep, objects):
    print(f"[time step {timestep}] ", end="")
    offsets = []
    for obj in objects:
        # 중심선 기준 거리 계산 (RoI 1: 5, RoI 2: 20, RoI 3: 40)
        offset = obj.calculate_offset()  # 객체 중심까지의 거리
        roi_offsets = (round(abs(offset - 5), 2), round(abs(offset - 20), 2), round(abs(offset - 40), 2))  # RoI별 거리
        offsets.append(roi_offsets)

    # 출력
    for i, roi_offsets in enumerate(offsets):
        print(f"object {i + 1} offset {roi_offsets} ", end="")
        if i < len(offsets) - 1:
            print("/ ", end="")

    return offsets


class IMM():
    def __init__(self, num_models, transition_matrix, initial_mu, initial_states, initial_covariances):
        """Initilize"""
        self.num_models = num_models    # 3: RoI 1, RoI 2, RoI 3
        self.transition_matrix = np.array(transition_matrix)    # 3x3
        self.mu = np.array(initial_mu)  # 3x1
        self.states = initial_states  # 각 객체의 초기 상태(랜덤하게 받은거 그대로 먹이기)
        self.covariances = initial_covariances  # 공분산 설정

    def interaction_step(self):
        """Model interaction: Calculate mixed initial state and covariance."""
        mixed_states = []
        mixed_covariances = []
        mu_conditional = self.transition_matrix.T @ self.mu
        c_bar = mu_conditional / np.sum(mu_conditional)

        for i in range(self.num_models):
            mixed_state = np.sum(
                [self.transition_matrix[j, i] * self.states[j] for j in range(self.num_models)], axis=0
            )
            mixed_states.append(mixed_state)
            mixed_covariance = np.sum(
                [self.transition_matrix[j, i] *
                 (self.covariances[j] + np.outer(self.states[j] - mixed_state, self.states[j] - mixed_state))
                 for j in range(self.num_models)],
                axis=0
            )
            mixed_covariances.append(mixed_covariance)

        return c_bar, mixed_states, mixed_covariances

    def prediction_step(self, mixed_states, mixed_covariances, process_models, process_covariances):
        """Model-specific prediction step."""
        predicted_states = []
        predicted_covariances = []
        for i in range(self.num_models):
            A = process_models[i]
            Q = process_covariances[i]
            predicted_state = A @ mixed_states[i]
            predicted_covariance = A @ mixed_covariances[i] @ A.T + Q
            predicted_states.append(predicted_state)
            predicted_covariances.append(predicted_covariance)

        return predicted_states, predicted_covariances

    def update_step(self, observations, observation_models, observation_covariances):
        """Model-specific update step."""
        updated_states = []
        updated_covariances = []
        likelihoods = np.zeros(self.num_models)

        for i in range(self.num_models):
            H = observation_models[i]
            R = observation_covariances[i]
            z = observations
            y = z - H @ self.states[i]
            S = H @ self.covariances[i] @ H.T + R
            K = self.covariances[i] @ H.T @ np.linalg.inv(S)
            updated_state = self.states[i] + K @ y
            updated_covariance = (np.eye(len(self.states[i])) - K @ H) @ self.covariances[i]

            likelihoods[i] = np.exp(-0.5 * y.T @ np.linalg.inv(S) @ y) / np.sqrt(np.linalg.det(2 * np.pi * S))

            updated_states.append(updated_state)
            updated_covariances.append(updated_covariance)

        return updated_states, updated_covariances, likelihoods

    def fusion_step(self, likelihoods):
        """IMM model fusion."""
        self.mu = likelihoods * self.mu
        self.mu /= np.sum(self.mu)  # Normalize

        fused_state = np.sum([self.mu[i] * self.states[i] for i in range(self.num_models)], axis=0)
        fused_covariance = np.sum(
            [self.mu[i] * (self.covariances[i] + np.outer(self.states[i] - fused_state, self.states[i] - fused_state))
             for i in range(self.num_models)],
            axis=0
        )

        return fused_state, fused_covariance






def main():
    init_settings()
    screen, clock = init_pygame()

    # 객체 생성 (ID 부여)
    objects = [DynamicObject(id=i + 1) for i in range(3)]
    last_log_time = time.time()
    timestep = 1  # Time step counter

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
            obs_offsets = log_offsets(timestep, objects)
            timestep += 1

            log_object_states(objects)
            last_log_time = current_time
            # print(obs_offsets)    # 이걸로 IMM 동작.
            print()


        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# 실행
if __name__ == "__main__":
    main()
