from typing import List, Tuple, Union, Literal

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Attention layer"""

    def __init__(self, input_dim: int, temperature: float = 1.0) -> None:
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.temperature = temperature
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        alignment = self.linear(x).squeeze(dim=-1)  # (n, 1) -> (n, )
        attention_weight = torch.softmax(alignment / self.temperature, dim=0)  # (n,)
        return attention_weight


class AttentionMIL(nn.Module):
    """인코더를 입력받는 기본 Attention MIL 모델"""

    def __init__(
        self, encoder, encoder_dim: int, adaptor_dim=int, num_classes: int = 2
    ):
        super(AttentionMIL, self).__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.adaptor_dim = adaptor_dim
        self.num_classes = num_classes

        # Freezing the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Adding an adaptor layer
        self.adaptor = nn.Sequential(
            nn.Linear(encoder_dim, adaptor_dim),
            nn.ReLU(),
            nn.Linear(adaptor_dim, adaptor_dim),
        )
        self.attention_layer = AttentionLayer(adaptor_dim)
        self.classifier = nn.Linear(adaptor_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, C, H, W)

        Returns:
            torch.Tensor: _description_
        """
        if x.ndim == 5:
            x = x.squeeze(0)  # (N, C, H, W)

        n_instance, C, H, W = x.shape

        instance_features = self.encoder(x)
        instance_features = instance_features.view(n_instance, -1)  # (N, feature)
        instance_features = self.adaptor(instance_features)
        attention_weights = self.attention_layer(instance_features)  # (N,)
        weighted_features = torch.einsum(
            "i,ij->ij", attention_weights, instance_features
        )

        context_vector = weighted_features.sum(axis=0)
        logit = self.classifier(context_vector).unsqueeze(dim=0)
        # instance_contribution = torch.einsum(
        #     "i,ij->ij", attention_weights, self.classifier(instance_features).squeeze(1)
        # )
        return logit, attention_weights


class Adaptor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Dropout 추가
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = self.proj(x) if self.proj else x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout 적용
        x = self.fc2(x)
        return self.norm(x + residual)


class AttentionBasedFeatureMIL(nn.Module):
    """Feature로부터 forward하는 Attention MIL 모델"""

    def __init__(
        self,
        in_features: int,
        adaptor_dim: int = 256,
        num_classes: int = 2,
        temperature: float = 1.0,
        threshold: float = None,
        return_with: Literal[
            "contribution", "attention_weight", "attention_score"
        ] = "attention_score",
        **kwargs
    ):
        super(AttentionBasedFeatureMIL, self).__init__()
        self.in_features = in_features
        self.adaptor_dim = adaptor_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.threshold = threshold
        self.return_with = return_with

        # Adding an adaptor layer
        self.adaptor = nn.Sequential(
            nn.Linear(in_features, adaptor_dim),
            nn.ReLU(),
            nn.Linear(adaptor_dim, in_features),
        )
        self.attention_layer = AttentionLayer(in_features, self.temperature)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): (1, N, D)

        Returns:
            torch.Tensor: _description_
        """
        if x.ndim == 3:
            x = x.squeeze(0)

        instance_features = self.adaptor(x)
        alignment = self.attention_layer.linear(instance_features).squeeze(
            dim=-1
        )  # (N, )

        attention_weights = torch.softmax(alignment / self.temperature, dim=0)  # (n,)

        if self.threshold is not None:
            n_patches = attention_weights.size(0)
            # threshold is not None인 경우 threshold 처리
            thresholded = attention_weights - (self.threshold / n_patches)
            thresholded = torch.clamp(thresholded, min=0.0)
            nom = thresholded.sum() + 1e-8

            attention_weights = thresholded / nom

        weighted_features = torch.einsum(
            "i,ij->ij", attention_weights, instance_features
        )

        context_vector = weighted_features.sum(axis=0)
        logit = self.classifier(context_vector)

        if self.return_with == "contribution":
            instance_contribution = attention_weights * self.classifier(
                instance_features
            ).squeeze(1)
            return logit, instance_contribution

        if self.return_with == "attention_weight":
            return logit, attention_weights

        if self.return_with == "attention_score":
            return logit, alignment

        return logit, attention_weights


class Attn_Net(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int, dropout: float = 0.0, n_classes: int = 1
    ):
        super(Attn_Net, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (N, input_dim)
        return self.net(x), x


class Attn_Net_Gated(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int, dropout: float = 0.0, n_classes: int = 1
    ):
        super(Attn_Net_Gated, self).__init__()
        # branch a: Tanh 활성화
        a_layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        # branch b: Sigmoid 활성화
        b_layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        if dropout > 0:
            a_layers.append(nn.Dropout(dropout))
            b_layers.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*a_layers)
        self.attention_b = nn.Sequential(*b_layers)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a * b  # element-wise multiplication
        A = self.fc(A)
        return A, x


class CLAM_SB(nn.Module):
    """
    Feature 기반 CLAM_SB 모델.

    인자:
      - in_features: 입력 feature 차원
      - adaptor_dim: adaptor 네트워크의 출력 차원
      - num_classes: 분류할 클래스 수
      - dropout: dropout 확률
      - k_sample: instance–level 학습 시 선택할 상위/하위 샘플 개수
      - gate: gated attention 사용 여부
      - instance_loss_fn: 인스턴스 분류기에 사용할 손실 함수 (기본: CrossEntropyLoss)
    """

    def __init__(
        self,
        in_features: int,
        adaptor_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.0,
        k_sample: int = 8,
        gate: bool = True,
        instance_loss_fn=nn.CrossEntropyLoss(),
    ):
        super(CLAM_SB, self).__init__()
        self.in_features = in_features
        # Adaptor 네트워크: 간단한 fc + ReLU + Dropout
        self.adaptor = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(dropout)
        )
        # attention 네트워크 (출력 차원 1)
        if gate:
            self.attention_net = Attn_Net_Gated(
                input_dim=512,
                hidden_dim=adaptor_dim,
                dropout=dropout,
                n_classes=1,
            )
        else:
            self.attention_net = Attn_Net(
                input_dim=512,
                hidden_dim=adaptor_dim,
                dropout=dropout,
                n_classes=1,
            )
        # bag–level 분류기
        self.classifier = nn.Linear(512, num_classes)
        # 인스턴스–level 분류기 (클래스별로 2–클래스 분류)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(512, 2) for _ in range(num_classes)]
        )
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.num_classes = num_classes

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, dtype=torch.long, device=device)

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, dtype=torch.long, device=device)

    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        # instance 수 조절
        N = A.size(1)
        k = min(self.k_sample, int(N / 2))

        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        """
        instance–level evaluation (out–of–class)
        """
        device = h.device
        A = A.view(-1)
        top_p_ids = torch.topk(A, self.k_sample)[1]
        top_p = torch.index_select(h, 0, top_p_ids)
        n_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        preds = torch.argmax(logits, dim=1)
        inst_loss = self.instance_loss_fn(logits, n_targets)
        return inst_loss, preds, n_targets

    def forward(
        self,
        x,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        """
        Args:
          - x: 입력 feature, shape (N, in_features)
          - label: bag 레벨 정답 (정수, 0 또는 1)
          - instance_eval: instance–level 평가 수행 여부
          - return_features: bag–level feature 반환 여부
          - attention_only: attention map만 반환 (디버깅용)
        Returns:
          (logits, Y_prob, Y_hat, raw attention map, 추가 결과 dict)
        """
        # 입력 feature를 adaptor 통과
        h = self.adaptor(x)  # (N, 512)
        # attention network 적용
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = torch.nn.functional.softmax(A, dim=1)  # softmax over N

        # instance-level 평가 (선택적)
        if instance_eval and label is not None:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    inst_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    total_inst_loss += inst_loss

        # bag–level feature: weighted sum over 인스턴스
        M = torch.mm(A, h)  # (1, adaptor_dim)
        logits = self.classifier(M)  # (1, num_classes)
        Y_prob = torch.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1, keepdim=True)
        results_dict = {}
        if instance_eval and label is not None:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_preds": all_preds,
                "inst_targets": all_targets,
            }
        if return_features:
            results_dict["features"] = M
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    """
    Feature 기반 CLAM_MB 모델.
    """

    def __init__(
        self,
        in_features: int,
        adaptor_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.0,
        k_sample: int = 8,
        gate: bool = True,
        instance_loss_fn=nn.CrossEntropyLoss(),
    ):
        super(CLAM_MB, self).__init__(
            in_features=in_features,
            adaptor_dim=adaptor_dim,
            num_classes=num_classes,
            dropout=dropout,
            k_sample=k_sample,
            gate=gate,
            instance_loss_fn=instance_loss_fn,
        )
        self.in_features = in_features
        self.adaptor = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(dropout)
        )
        # attention 네트워크: n_classes 개의 출력 (각 클래스별 attention map)
        if gate:
            self.attention_net = Attn_Net_Gated(
                input_dim=512,
                hidden_dim=adaptor_dim,
                dropout=dropout,
                n_classes=num_classes,
            )
        else:
            self.attention_net = Attn_Net(
                input_dim=512,
                hidden_dim=adaptor_dim,
                dropout=dropout,
                n_classes=num_classes,
            )
        # 각 클래스별 bag classifier (feature에서 logit을 산출)
        self.bag_classifiers = nn.ModuleList(
            [nn.Linear(512, 1) for _ in range(num_classes)]
        )
        # instance classifier는 CLAM_SB와 같이 각 클래스마다 정의
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(512, 2) for _ in range(num_classes)]
        )
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.num_classes = num_classes

    def forward(
        self,
        x,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        """
        Args:
          - x: 입력 feature, shape (N, in_features)
          - label: bag 레벨 정답 (정수)
          - instance_eval: instance–level 평가 수행 여부
          - return_features: bag–level feature 반환 여부
          - attention_only: attention map만 반환
        """
        h = self.adaptor(x)  # (N, adaptor_dim)
        A, h = self.attention_net(h)  # A: (N, num_classes)
        A = A.transpose(0, 1)  # (num_classes, N)
        if attention_only:
            return A
        A_raw = A
        A = torch.softmax(A, dim=1)  # 각 클래스별로 softmax over instances

        if instance_eval and label is not None:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = torch.zeros(self.num_classes, device=x.device)
            inst_labels[label] = 1
            for i, classifier in enumerate(self.instance_classifiers):
                if inst_labels[i] == 1:
                    inst_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_targets.extend(targets.cpu().numpy().tolist())
                    total_inst_loss += inst_loss

        # 각 클래스별로 bag–level feature 산출: weighted sum
        M = torch.mm(A, h)  # (num_classes, adaptor_dim)
        logits = torch.zeros(1, self.num_classes, device=M.device)
        for c in range(self.num_classes):
            logits[0, c] = self.bag_classifiers[c](M[c])
        Y_prob = torch.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1, keepdim=True)
        results_dict = {}
        if instance_eval and label is not None:
            results_dict = {
                "instance_loss": total_inst_loss,
                "inst_preds": all_preds,
                "inst_targets": all_targets,
            }
        if return_features:
            results_dict["features"] = M
        return logits, Y_prob, Y_hat, A_raw, results_dict
