# drl/agent/recurrent_sac_inference.py
from typing import Optional, Tuple

import torch

from drl.agent.agent import Agent, agent_config
from drl.agent.net import RecurrentSACNetwork # Use the main SAC network interface
from drl.exp import Experience
from drl.net import Network # Base network class

@agent_config(name="Recurrent SAC Inference")
class RecurrentSACInference(Agent):
    """
    Inference agent for Recurrent SAC. Uses the deterministic action (mean).
    """
    def __init__(
        self,
        network: RecurrentSACNetwork, # Expects the full network, will only use actor part
        num_envs: int,
        device: Optional[str] = None
    ) -> None:
        # Initialize Agent base class. Pass the full network,
        # but internally we'll primarily use its actor forward pass.
        super().__init__(num_envs, network, device)
        self._network = network

        # Hidden state tracking for the actor
        actor_hidden_shape = network.hidden_state_shape() # Assuming actor/critic use same shape
        self._actor_hidden_state = torch.zeros((num_envs,) + actor_hidden_shape, device=self.device)
        self._next_actor_hidden_state = torch.zeros((num_envs,) + actor_hidden_shape, device=self.device)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """ Select action deterministically (using mode/mean) """
        # Reset hidden state if previous step was terminal
        actor_h = self._next_actor_hidden_state * (1.0 - self._prev_terminated.view(-1, 1, 1)) # Adjust view based on hidden state dims

        # Ensure hidden state has correct shape (e.g., (layers*D, batch, H))
        actor_h = actor_h.permute(1, 0, 2) # Example: (batch, layers*D, H) -> (layers*D, batch, H)

        # Add sequence dimension (seq_len=1)
        obs_seq = obs.unsqueeze(1)

        # Forward pass through the actor part of the network
        policy_dist, next_actor_h = self._network.forward_actor(obs_seq, actor_h)

        # Select the mode (most likely action) for deterministic inference
        action = policy_dist.mode() # Shape: (batch_size, seq_len, action_dim)

        # Store next hidden state for the next step
        self._next_actor_hidden_state = next_actor_h.permute(1, 0, 2) # Back to (batch, layers*D, H)

        # Remove sequence dimension
        return action.squeeze(1)

    def update(self, exp: Experience) -> Optional[dict]:
        """ Inference agent does not learn, only updates hidden state """
        # Update internal state based on termination, needed for select_action
        self._prev_terminated = exp.terminated.clone()
        return None # No learning update

    def inference_agent(self, num_envs: int = 1, device: Optional[str] = None) -> Agent:
        """ Returns itself, as it's already an inference agent """
        # If num_envs or device changes, create a new instance
        if num_envs != self.num_envs or (device and device != str(self.device)):
             inference_device = device or str(self.device)
             return RecurrentSACInference(self._network, num_envs, inference_device)
        return self
