import numpy as np
import matplotlib.pyplot as plt
import os

def plot_time_series(time_steps, data, ylabel, title, legend_labels=None):
    """
    time_steps: 1D array of time steps (예: np.arange(n_steps))
    data: 2D array, shape (num_objects, n_steps)
    ylabel: y축 라벨 (예: "Offset (m)")
    title: 그래프 제목
    legend_labels: 객체별 레전드 라벨 (옵션)
    """
    num_objects = data.shape[0]
    plt.figure()
    for i in range(num_objects):
        label = f"Object {i}" if legend_labels is None else legend_labels[i]
        plt.plot(time_steps, data[i, :], marker='o', label=label)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_multiple_series(time_steps, current_data, predicted_data, ylabel, title, legend_labels=None):
    """
    current_data: 2D array (num_objects, n_steps)
    predicted_data: 2D array (num_objects, n_steps)
    두 시리즈를 한 그래프에 같이 그려 비교할 수 있습니다.
    """
    num_objects = current_data.shape[0]
    plt.figure()
    for i in range(num_objects):
        label_current = f"Object {i} Current" if legend_labels is None else legend_labels[i] + " Current"
        label_pred = f"Object {i} Predicted" if legend_labels is None else legend_labels[i] + " Predicted"
        plt.plot(time_steps, current_data[i, :], marker='o', linestyle='-', label=label_current)
        plt.plot(time_steps, predicted_data[i, :], marker='x', linestyle='--', label=label_pred)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_risks(time_steps, curr_risks, pred_risks):
    """
    curr_risks: 2D array (num_objects, n_steps) - 현재 위험도
    pred_risks: 2D array (num_objects, n_steps) - 예측 위험도
    """
    num_objects = curr_risks.shape[0]
    plt.figure()
    for i in range(num_objects):
        plt.plot(time_steps, curr_risks[i, :], marker='o', label=f"Object {i} Current Risk")
        plt.plot(time_steps, pred_risks[i, :], marker='x', linestyle='--', label=f"Object {i} Predicted Risk")
    plt.xlabel("Time Step")
    plt.ylabel("Risk")
    plt.title("Risk Assessment Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_offsets(max_steps, offsets, pred_offsets, n_steps):
    """
    현재 offset과 예측 offset을 시각적으로 비교하는 함수.

    Parameters:
    - max_steps: 총 시뮬레이션 스텝 수
    - offsets: shape (num_objects, max_steps) - 현재 offset 값
    - pred_offsets: shape (num_objects, max_steps, n_steps) - n_steps만큼의 예측 offset 값
    - n_steps: 미래 예측 스텝 수 (1 이상)

    - 각 객체별로 현재 offset과, n_steps 개수만큼의 미래 예측 offset을 다른 색상과 스타일로 표현.
    """
    num_objects = offsets.shape[0]  # 동적 객체 개수
    time_steps = np.arange(max_steps)  # (max_steps,) 형태의 1D 배열로 변환

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'cyan', 'magenta']

    plt.figure(figsize=(12, 6))

    for i in range(num_objects):
        # 현재 offset을 실선으로 표시
        plt.plot(time_steps, offsets[i, :], label=f'Object {i} Current Offset',
                 color=colors[i % len(colors)], linestyle='-')

        # # n_steps 개수만큼의 미래 예측 값을 점선으로 표시
        # for step in range(n_steps):
        #     plt.plot(time_steps, pred_offsets[i, :, step],
        #              label=f'Object {i} Predicted Offset (Step {step + 1})',
        #              color=colors[i % len(colors)], linestyle='--', alpha=0.6)

        # (진우수정) 예측값 1타임씩 밀리게
        time_steps = np.arange(max_steps)
        # 예측 값은 1, 2, 3, ... max_steps (즉, 한 타임 스텝 뒤로 밀림)
        pred_time_steps = np.arange(1, max_steps + 1)
        for step in range(n_steps):
            plt.plot(pred_time_steps, pred_offsets[i, :, step],
                     label=f'Object {i} Predicted Offset (Step {step + 1})',
                     color=colors[i % len(colors)], linestyle='--', alpha=0.6)


    plt.xticks(np.arange(0, max_steps+1, 10))
    plt.xlabel('Time Step')
    plt.ylabel('Offset (m)')
    plt.ylim((0,12))
    plt.title(f'Current vs {n_steps}-Step Predicted Offsets')
    plt.legend()
    plt.grid(True)
    plt.show()


# (진우 수정3) 그래프 저장하는 함수.
def save_plot_offsets(max_steps, offsets, n_steps_pred, filename):
    """
    현재 offset과 예측 offset을 시각적으로 비교하는 함수.

    Parameters:
    - max_steps: 총 시뮬레이션 스텝 수
    - offsets: shape (num_objects, max_steps) - 현재 offset 값
    - pred_offsets: shape (num_objects, max_steps, n_steps) - n_steps만큼의 예측 offset 값
    - n_steps: 미래 예측 스텝 수 (1 이상)
    - filename: 저장할 그래프 이름.

    - 각 객체별로 현재 offset과, n_steps 개수만큼의 미래 예측 offset을 다른 색상과 스타일로 표현.
    """

    # 저장할 디렉토리 생성
    save_dir = "Offset_plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    num_objects = offsets.shape[0]  # 동적 객체 개수
    time_steps = np.arange(max_steps)  # (max_steps,) 형태의 1D 배열로 변환

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'cyan', 'magenta']

    plt.figure(figsize=(12, 6))

    for obj in range(num_objects):
        # 실제 위치를 연결한 실선 (모든 timestep의 현재 위치: offsets[obj, :, 0])
        time_steps = np.arange(max_steps)
        actual_positions = offsets[obj, :max_steps, 0]
        plt.plot(time_steps, actual_positions, linestyle='-', marker='o', ms = 3,
                 color=colors[obj % len(colors)], label=f"Object {obj} Actual")

        # 각 timestep별로 예측 값을 점선으로 표시하고, 개별 점들은 점으로 찍음
        for step in range(max_steps):
            # 현재 timestep에서의 현재 위치와 예측값: indices 0 ~ n_steps_pred
            pred_vals = offsets[obj, step, :n_steps_pred + 1]
            # x축은 현재 timestep부터 (step, step+1, ..., step+n_steps_pred)
            x_vals = step + np.arange(n_steps_pred + 1)

            # 점선으로 연결 (예측 궤적)
            plt.plot(x_vals, pred_vals, linestyle='--',
                     color=colors[obj % len(colors)], alpha=0.5)

    #
    # for i in range(num_objects):
    #     # 현재 offset을 실선으로 표시
    #     plt.plot(time_steps, offsets[i, :], label=f'Object {i} Current Offset',
    #              color=colors[i % len(colors)], linestyle='-')
    #
    #     # # n_steps 개수만큼의 미래 예측 값을 점선으로 표시
    #     # for step in range(n_steps):
    #     #     plt.plot(time_steps, pred_offsets[i, :, step],
    #     #              label=f'Object {i} Predicted Offset (Step {step + 1})',
    #     #              color=colors[i % len(colors)], linestyle='--', alpha=0.6)
    #
    #     # (진우수정) 예측값 1타임씩 밀리게
    #     time_steps = np.arange(max_steps)
    #     # 예측 값은 1, 2, 3, ... max_steps (즉, 한 타임 스텝 뒤로 밀림)
    #     pred_time_steps = np.arange(1, max_steps + 1)
    #     for step in range(n_steps):
    #         plt.plot(pred_time_steps, pred_offsets[i, :, step],
    #                  label=f'Object {i} Predicted Offset (Step {step + 1})',
    #                  color=colors[i % len(colors)], linestyle='--', alpha=0.6)


    plt.xticks(np.arange(0, max_steps+1, 10))
    plt.xlabel('Time Step')
    plt.ylabel('Offset (m)')
    plt.ylim((0, 12))
    plt.title(f'Current vs {n_steps_pred}-Step Predicted Offsets')
    plt.legend()
    plt.grid(True)

    # 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()