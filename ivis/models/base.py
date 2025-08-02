from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for IViS-compatible imaging models.
    All models must implement loss() and forward().
    """

    @abstractmethod
    def loss(self, x: np.ndarray, *args) -> tuple[float, np.ndarray]:
        """
        Compute scalar loss and gradient for optimization.
        
        Parameters
        ----------
        x : np.ndarray
            Flattened parameter vector.

        Returns
        -------
        loss : float
            Scalar loss.
        grad : np.ndarray
            Flattened gradient.
        """
        pass

    @abstractmethod
    def forward(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Simulate model visibilities from image parameters.

        Parameters
        ----------
        x : np.ndarray
            Sky model or parameters (flattened or shaped).

        Returns
        -------
        model_vis : np.ndarray
            Predicted complex visibilities.
        """
        pass
