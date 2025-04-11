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
