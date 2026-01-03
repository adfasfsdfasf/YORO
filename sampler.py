import abc
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import tqdm


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        # if not 0 < percentage < 1:
        #     raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)
    
class KMeansPlusPlusSampler(BaseSampler):
    def __init__(self, percentage: float, device: torch.device):
        """
        KMeans++ sampling base class.

        Args:
            percentage: Number of clusters to sample as a percentage of total features.
            device: Device to perform computations on (e.g., 'cpu' or 'cuda').
        """
        self.percentage = percentage
        self.device = device

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Subsamples features using KMeans++.

        Args:
            features: [N x D] input feature collection.

        Returns:
            Subsampled features of shape [percentage x D].
        """
        # Ensure features are in torch.Tensor format and on the correct device
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.device)
        else:
            features = features.to(self.device)

        num_clusters = int(self.percentage)
        num_points, feature_dim = features.shape

        # Step 1: Initialize the first cluster center randomly
        first_center_idx = torch.randint(0, num_points, (1,)).item()
        centers = features[first_center_idx].unsqueeze(0)  # Shape: [1, D]

        # Step 2: Iteratively select the remaining cluster centers
        for _ in range(1, num_clusters):
            print(f"Selecting center {_ + 1}/{num_clusters}...")
            # Compute distances from all points to the existing centers
            distances = torch.cdist(features, centers, p=2)  # Shape: [N, num_centers]
            min_distances, _ = torch.min(distances, dim=1)  # Shape: [N]

            # Compute probabilities proportional to the squared distances
            probabilities = min_distances ** 2
            probabilities /= torch.sum(probabilities)

            # Sample the next center based on the computed probabilities
            next_center_idx = torch.multinomial(probabilities, 1).item()
            next_center = features[next_center_idx].unsqueeze(0)  # Shape: [1, D]

            # Add the new center to the list of centers
            centers = torch.cat([centers, next_center], dim=0)  # Shape: [num_centers, D]

        return self._restore_type(centers)




class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        # if self.percentage == 1:
        #     return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
       matrix_a: torch.Tensor, matrix_b: torch.Tensor
   ) -> torch.Tensor:
       #Computes batchwise Euclidean distances using PyTorch.
       a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
       b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
       a_times_b = matrix_a.mm(matrix_b.T)

       return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()
    
    # @staticmethod
    # def _compute_batchwise_differences(
    #      matrix_a: torch.Tensor, matrix_b: torch.Tensor
    # ) -> torch.Tensor:
    #     """Computes batchwise cosine distances using PyTorch."""
    #     #     # 计算矩阵的范数
    #     norm_a = matrix_a.norm(dim=1, keepdim=True)  # [N, 1]
    #     norm_b = matrix_b.norm(dim=1, keepdim=True)  # [M, 1]

    #     # 计算点积
    #     dot_product = matrix_a.mm(matrix_b.T)  # [N, M]
    
    #     # 计算余弦相似度
    #     cosine_similarity = dot_product / (norm_a * norm_b.T)  # [N, M]
    
    # #     # 转换为余弦距离
    #     cosine_distance = 1 - cosine_similarity
    
    #     return cosine_distance



    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(self.percentage)
        num_coreset_samples = int(self.percentage)
        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)

import time
class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.

        Args:
            features: [N x D]
        """
        np.random.seed(int(time.time()))  # For reproducibility
        num_random_samples = int(self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]
