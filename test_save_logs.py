import pandas as pd
import numpy as np
from openpyxl import load_workbook

def save_risks(iter, max_steps, curr_Risks, pred_Risks):
    """
    현재 기반 위협지수, 예측 기반 위협지수 액셀 파일로 저장.
    :param iter:
    :param max_steps:
    :param curr_Risks:
    :param pred_Risks:
    :return:
    """
    df_risks = pd.DataFrame({
        "Step": np.arange(max_steps),
        "Object 0 - Current Risk": curr_Risks[0],
        "Object 1 - Current Risk": curr_Risks[1],
        "Object 2 - Current Risk": curr_Risks[2],
        "Object 0 - Predicted Risk": pred_Risks[0],
        "Object 1 - Predicted Risk": pred_Risks[1],
        "Object 2 - Predicted Risk": pred_Risks[2]
    })

    excel_filename = f"Logs/Logs_iter_{iter}.xlsx"


    df_risks.to_excel(excel_filename, index=False)

def save_actions(iter, random_based_actions, curr_based_actions, pred_based_actions, total_based_actions):
    """
    case 1,2,3,4의 액션 시퀀스 액셀에 추가
    :param iter:
    :param random_based_actions:
    :param curr_based_actions:
    :param pred_based_actions:
    :param total_based_actions:
    :return:
    """

    excel_filename = f"Logs/Logs_iter_{iter}.xlsx"

    # 1. 기존 엑셀 불러오기
    df_existing = pd.read_excel(excel_filename)

    df_actions = pd.DataFrame({
        "Random Action": random_based_actions,
        "Current-Based Action": curr_based_actions,
        "Predicted-Based Action": pred_based_actions,
        "Total-Based Action": total_based_actions,
    })

    # 몇 줄 아래에 쓸 건지 계산 (100줄짜리 risk가 위에 있다고 가정)
    df_merged = pd.concat([df_existing, df_actions], axis=1)

    # 4. 덮어쓰기 저장
    df_merged.to_excel(excel_filename, index=False)

def save_rewards(iter, random_based_reward, curr_based_reward, pred_based_reward, total_based_reward):
    """
    case 1,2,3,4의 리워드(10 스탭 이후의) 액셀에 추가
    :param iter:
    :param random_based_reward:
    :param curr_based_reward:
    :param pred_based_reward:
    :param total_based_reward:
    :return:
    """
    excel_filename = f"Logs/Logs_iter_{iter}.xlsx"

    rewards_row = pd.DataFrame({
        "Rewards sum after 10 steps":[""],
        "Random Reward": [random_based_reward],
        "Current Reward": [curr_based_reward],
        "Predicted Reward": [pred_based_reward],
        "Total Reward": [total_based_reward]
    })

    df_existing = pd.read_excel(excel_filename)
    df_merged = pd.concat([df_existing, rewards_row], axis=1)


    df_merged.to_excel(excel_filename, index=False)

def fit_exal(iter):
    """
    액셀 이쁘게
    """
    excel_filename = f"Logs/Logs_iter_{iter}.xlsx"
    # 열 너비 조절을 위해 openpyxl로 불러와서 조정
    wb = load_workbook(excel_filename)
    ws = wb.active

    # 각 열 제목 길이에 맞춰 너비 조정
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = max_length + 2  # 약간 여유
        ws.column_dimensions[col_letter].width = adjusted_width

    # 다시 저장
    wb.save(excel_filename)