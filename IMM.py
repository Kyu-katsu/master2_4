class IMM:
    def __init__(self, init_model_prob, init_distribution_var):
        self.model_num = 3  # 모델 개수 M
        self.mu = init_model_prob   # 모델 확률 \mu
        self.P = init_distribution_var  # 분포 분산 \mathbf{P}

    def mixed_prediction(self):
