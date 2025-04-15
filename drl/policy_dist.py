# drl/policy_dist.py
import torch
import torch.nn.functional as F
from abc import abstractmethod
from typing import Optional

class Distribution:
    @abstractmethod
    def sample(self) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def entropy(self) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def mode(self) -> torch.Tensor:
        raise NotImplementedError

class CategoricalDist(Distribution):
    """
    Categorical distribution based on logits.
    """
    def __init__(self, logits: torch.Tensor):
        # Ensure logits are float for stability with distributions
        self._logits = logits.float()
        # It's generally safer to create the distribution object as needed
        # rather than storing it, especially if logits change.
        # self.dist = torch.distributions.Categorical(logits=self._logits)

    def sample(self) -> torch.Tensor:
        # Create distribution on the fly
        dist = torch.distributions.Categorical(logits=self._logits)
        # Sample and add action dimension
        return dist.sample().unsqueeze(-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Create distribution on the fly
        dist = torch.distributions.Categorical(logits=self._logits)
        # Remove action dimension if present before calculating log_prob
        actions_squeezed = actions.squeeze(-1)
        # Calculate log_prob and add action dimension back
        return dist.log_prob(actions_squeezed).unsqueeze(-1)

    def entropy(self) -> torch.Tensor:
        # Create distribution on the fly
        dist = torch.distributions.Categorical(logits=self._logits)
        # Calculate entropy and add action dimension
        return dist.entropy().unsqueeze(-1)

    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    def mode(self) -> torch.Tensor:
        """Returns the action with the highest probability (logit)."""
        # Find the index of the max logit along the action dimension
        # Add an action dimension at the end
        return torch.argmax(self._logits, dim=-1, keepdim=True)

# Keep SACDistribution interface if needed for other distribution types later
class SACDistribution(Distribution):
    @abstractmethod
    def sample(self) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def entropy(self) -> torch.Tensor:
        raise NotImplementedError
    @abstractmethod
    def mode(self) -> torch.Tensor:
        raise NotImplementedError
