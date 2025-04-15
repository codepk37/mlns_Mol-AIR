# train/net.py
from typing import Tuple, Iterable, Optional # <<< Added Optional import here

import torch
import torch.nn as nn
import torch.nn.functional as F # <<< Added F import
import torch.nn.init as init

import drl.agent as agent_interfaces # Use interfaces from drl.agent.net
import drl.net as drl_net_base # Base classes like wrap/unwrap LSTM
from drl.policy import CategoricalPolicy
from drl.policy_dist import CategoricalDist

# --- Helper Functions ---
def init_linear_weights(model: nn.Module, gain: float = 2.0**0.5) -> None:
    """Initialize linear layers with orthogonal initialization."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            init.orthogonal_(layer.weight, gain)
            if layer.bias is not None:
                layer.bias.data.zero_()

# --- Shared Recurrent Backbone (Example using LSTM) ---
class SelfiesRecurrentSharedNet(nn.Module):
    """ A shared LSTM backbone for actor and critic networks. """
    def __init__(self, in_features: int, hidden_dim: int = 64, n_recurrent_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_recurrent_layers = n_recurrent_layers
        # Calculate combined hidden state dim for LSTM (H_cell + H_out)
        # Assuming H_cell = H_out = hidden_dim
        self.lstm_hidden_dim = hidden_dim * 2

        # Input embedding layer (e.g., for one-hot encoded tokens)
        # Maps input features (vocab size) to the LSTM's expected input dimension
        self.input_embedding = nn.Linear(in_features, hidden_dim)

        self.recurrent_layer = nn.LSTM(
            input_size=hidden_dim, # Input to LSTM is embedded features
            hidden_size=hidden_dim, # H_out size
            batch_first=True,       # Input/output tensors are (batch, seq, feature)
            num_layers=n_recurrent_layers
        )
        # Output features after LSTM will match hidden_dim (H_out)
        self.out_features = hidden_dim

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the shared recurrent backbone.

        Args:
            x (torch.Tensor): Input sequence (batch_size, seq_len, in_features).
            hidden_state (torch.Tensor): LSTM hidden state ((num_layers * D, batch_size, H_cell), (num_layers * D, batch_size, H_out)).
                                         Or wrapped as (num_layers * D, batch_size, H_cell + H_out).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output sequence from LSTM (batch_size, seq_len, hidden_dim).
                - Next LSTM hidden state ((num_layers * D, batch_size, H_cell), (num_layers * D, batch_size, H_out)) wrapped.
        """
        # Apply input embedding
        # Input x shape: (batch_size, seq_len, in_features)
        embedded_seq = F.relu(self.input_embedding(x)) # Shape: (batch_size, seq_len, hidden_dim)

        # Feed forward to the recurrent layers
        # Unwrap the combined hidden state into h_0 and c_0 for LSTM
        h, c = drl_net_base.unwrap_lstm_hidden_state(hidden_state, h_size=self.hidden_dim, c_size=self.hidden_dim)
        # LSTM expects hidden state tuple: (h_0, c_0)
        # h_0 shape: (num_layers * D, batch_size, H_out)
        # c_0 shape: (num_layers * D, batch_size, H_cell)
        lstm_out_seq, (h_n, c_n) = self.recurrent_layer(embedded_seq, (h, c))
        # lstm_out_seq shape: (batch_size, seq_len, hidden_dim)

        # Wrap the next hidden state (h_n, c_n) back into a single tensor
        next_seq_hidden_state = drl_net_base.wrap_lstm_hidden_state(h_n, c_n)
        # next_seq_hidden_state shape: (num_layers * D, batch_size, H_cell + H_out)

        return lstm_out_seq, next_seq_hidden_state

    def hidden_state_shape(self) -> Tuple[int, int]:
         """ Returns (num_layers * D, H_cell + H_out) """
         # Assuming D=1 (unidirectional LSTM)
         return (self.n_recurrent_layers, self.lstm_hidden_dim)

# --- PPO/RND Networks (Keep them) ---
# Renamed internal class to avoid potential conflicts if used elsewhere
class SelfiesRecurrentPPOSharedNetInternal(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.hidden_state_dim = 64 * 2
        self.n_recurrent_layers = 2
        self.out_features = 256
        self.recurrent_layers = nn.LSTM(in_features, self.hidden_state_dim // 2, batch_first=True, num_layers=self.n_recurrent_layers)
        self.linear_layers = nn.Sequential(
            nn.Linear(self.hidden_state_dim // 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.out_features), nn.ReLU()
        )
    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = drl_net_base.unwrap_lstm_hidden_state(hidden_state)
        embedding_seq, (h_n, c_n) = self.recurrent_layers(x, (h, c))
        next_seq_hidden_state = drl_net_base.wrap_lstm_hidden_state(h_n, c_n)
        embedding_seq = self.linear_layers(embedding_seq)
        return embedding_seq, next_seq_hidden_state
    def hidden_state_shape(self) -> Tuple[int, int]:
        # Assuming D=1
        return (self.n_recurrent_layers, self.hidden_state_dim)


class SelfiesEmbeddedConcatRND(nn.Module):
    def __init__(self, obs_features, hidden_state_shape) -> None:
        super().__init__()
        # hidden_state_shape is expected as (num_layers * D, H)
        hidden_state_features = hidden_state_shape[0] * hidden_state_shape[1]
        self._predictor_obs_embedding = nn.Sequential(nn.Linear(obs_features, 64), nn.ReLU())
        self._predictor = nn.Sequential(
            nn.Linear(64 + hidden_state_features, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256)
        )
        self._target_obs_embedding = nn.Sequential(nn.Linear(obs_features, 64), nn.ReLU())
        self._target = nn.Sequential(
            nn.Linear(64 + hidden_state_features, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256)
        )
        for param in self._target_obs_embedding.parameters(): param.requires_grad = False
        for param in self._target.parameters(): param.requires_grad = False
    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_state input shape: (batch_size, num_layers * D * H) - needs flattening before concat
        hidden_state_flat = hidden_state.flatten(1)
        obs_embedding_pred = self._predictor_obs_embedding(obs)
        predicted_features = self._predictor(torch.cat([obs_embedding_pred, hidden_state_flat], dim=1))

        obs_embedding_target = self._target_obs_embedding(obs)
        target_features = self._target(torch.cat([obs_embedding_target, hidden_state_flat], dim=1))
        return predicted_features, target_features


class SelfiesPretrainedNet(nn.Module, agent_interfaces.PretrainedRecurrentNetwork):
     def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # Use the internal PPO shared net structure for pretraining
        self._shared_net = SelfiesRecurrentPPOSharedNetInternal(vocab_size)
        self._actor = CategoricalPolicy(self._shared_net.out_features, vocab_size)
        init_linear_weights(self) # Initialize weights

     def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[CategoricalDist, torch.Tensor]:
        # Pass through shared backbone
        embedding_seq, next_seq_hidden_state = self._shared_net(x, hidden_state)
        # Pass features through actor head
        policy_dist_seq = self._actor(embedding_seq)
        return policy_dist_seq, next_seq_hidden_state

     def hidden_state_shape(self) -> Tuple[int, int]:
        return self._shared_net.hidden_state_shape()

     def model(self) -> nn.Module:
        return self


class SelfiesRecurrentPPONet(nn.Module, agent_interfaces.RecurrentPPONetwork):
     def __init__(self, in_features: int, num_actions: int) -> None:
        super().__init__()
        # Use the internal PPO shared net structure
        self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNetInternal(in_features)
        self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, num_actions)
        self._critic = nn.Linear(self._actor_critic_shared_net.out_features, 1) # Critic head
        init_linear_weights(self) # Initialize weights

     def model(self) -> nn.Module:
        return self

     def hidden_state_shape(self) -> Tuple[int, int]:
        return self._actor_critic_shared_net.hidden_state_shape()

     def forward(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor]:
        # Pass through shared backbone
        embedding_seq, next_seq_hidden_state = self._actor_critic_shared_net(obs_seq, hidden_state)
        # Pass features through actor and critic heads
        policy_dist_seq = self._actor(embedding_seq)
        state_value_seq = self._critic(embedding_seq)
        return policy_dist_seq, state_value_seq, next_seq_hidden_state


class SelfiesRecurrentPPORNDNet(nn.Module, agent_interfaces.RecurrentPPORNDNetwork):
     def __init__(self, in_features: int, num_actions: int, temperature: float = 1.0) -> None:
        super().__init__()
        # Use the internal PPO shared net structure
        self._actor_critic_shared_net = SelfiesRecurrentPPOSharedNetInternal(in_features)
        self._actor = CategoricalPolicy(self._actor_critic_shared_net.out_features, num_actions, temperature=temperature)
        # Separate critics for extrinsic and intrinsic values
        self._ext_critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
        self._int_critic = nn.Linear(self._actor_critic_shared_net.out_features, 1)
        # RND predictor/target networks
        self._rnd_net = SelfiesEmbeddedConcatRND(in_features, self.hidden_state_shape())
        init_linear_weights(self) # Initialize weights

     def model(self) -> nn.Module:
        return self

     def hidden_state_shape(self) -> Tuple[int, int]:
        return self._actor_critic_shared_net.hidden_state_shape()

     def forward_actor_critic(self, obs_seq: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pass through shared backbone
        embedding_seq, next_seq_hidden_state = self._actor_critic_shared_net(obs_seq, hidden_state)
        # Pass features through actor and critic heads
        policy_dist_seq = self._actor(embedding_seq)
        ext_state_value_seq = self._ext_critic(embedding_seq)
        int_state_value_seq = self._int_critic(embedding_seq)
        return policy_dist_seq, ext_state_value_seq, int_state_value_seq, next_seq_hidden_state

     def forward_rnd(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: hidden_state here is expected to be flattened for RND network input
        return self._rnd_net(obs, hidden_state)


# --- New SAC Network Implementation ---
class SelfiesRecurrentSACNet(nn.Module, agent_interfaces.RecurrentSACNetwork):
    """
    Recurrent SAC Network implementation for SELFIES environment.
    Uses a shared LSTM backbone.
    """
    def __init__(self, obs_shape: int, num_actions: int, hidden_dim: int = 64, n_recurrent_layers: int = 2, temperature: float = 1.0):
        """
        Args:
            obs_shape (int): The size of the vocabulary (input features are one-hot encoded).
            num_actions (int): The number of discrete actions (vocabulary size).
            hidden_dim (int): The hidden dimension for LSTM and linear layers.
            n_recurrent_layers (int): The number of LSTM layers.
            temperature (float): Temperature for policy sampling (can be used in CategoricalPolicy).
        """
        super().__init__()
        self._obs_shape_val = obs_shape # Store vocab size for embedding
        self._num_actions = num_actions
        self._hidden_dim = hidden_dim
        self._n_recurrent_layers = n_recurrent_layers

        # Shared recurrent backbone
        self._shared_net = SelfiesRecurrentSharedNet(
            in_features=obs_shape, # Input is one-hot vocab size
            hidden_dim=hidden_dim,
            n_recurrent_layers=n_recurrent_layers
        )

        # --- Actor Head ---
        # Takes features from shared net and outputs action distribution logits
        self._actor_head = CategoricalPolicy(
            in_features=self._shared_net.out_features,
            num_discrete_actions=num_actions,
            temperature=temperature # Pass temperature if needed
        )

        # --- Critic Heads (Two Q-networks) ---
        # Takes features from shared net AND action embedding, outputs Q-values
        # Embed the discrete action to concatenate it with the LSTM output features
        self._action_embedding_dim = 32 # Example dimension for action embedding
        self._action_embed = nn.Embedding(num_embeddings=num_actions, embedding_dim=self._action_embedding_dim)

        # Input dimension for the critic heads
        critic_input_dim = self._shared_net.out_features + self._action_embedding_dim

        # Define the two critic heads
        self._critic1_head = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output single Q-value
        )
        self._critic2_head = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output single Q-value
        )

        # Initialize weights for all layers in this network
        init_linear_weights(self)

    def model(self) -> nn.Module:
        """Returns the entire network module."""
        return self

    def hidden_state_shape(self) -> Tuple[int, int]:
        """ Returns the shape required by the recurrent layer (LSTM). """
        # Shape is (num_layers * D, H_cell + H_out)
        return self._shared_net.hidden_state_shape()

    def forward_actor(
        self,
        obs_seq: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None # Make hidden_state optional for flexibility
    ) -> Tuple[CategoricalDist, torch.Tensor]:
        """ Actor forward pass. """
        if hidden_state is None:
             # Initialize hidden state if not provided (e.g., for batch size > 1 at step 0)
             batch_size = obs_seq.size(0)
             h_shape = self.hidden_state_shape()
             # Ensure hidden state is created on the correct device
             hidden_state = torch.zeros(h_shape[0], batch_size, h_shape[1], device=obs_seq.device)

        # Pass through shared backbone
        # obs_seq shape: (batch_size, seq_len, obs_shape)
        shared_features_seq, next_hidden_state = self._shared_net(obs_seq, hidden_state)
        # shared_features_seq shape: (batch_size, seq_len, hidden_dim)

        # Pass features through actor head
        policy_dist_seq = self._actor_head(shared_features_seq)

        return policy_dist_seq, next_hidden_state

    def forward_critic(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None # Make hidden_state optional
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Critic forward pass. """
        if hidden_state is None:
             # Initialize hidden state if not provided
             batch_size = obs_seq.size(0)
             h_shape = self.hidden_state_shape()
              # Ensure hidden state is created on the correct device
             hidden_state = torch.zeros(h_shape[0], batch_size, h_shape[1], device=obs_seq.device)

        # Pass observations through shared backbone
        shared_features_seq, next_hidden_state = self._shared_net(obs_seq, hidden_state)
        # shared_features_seq shape: (batch_size, seq_len, hidden_dim)

        # Embed actions
        # action_seq shape: (batch_size, seq_len, 1) or (batch_size, seq_len)
        # Ensure action_seq is LongTensor and remove trailing dim if present before embedding
        action_seq_long = action_seq.squeeze(-1).long()
        action_features_seq = self._action_embed(action_seq_long)
        # action_features_seq shape: (batch_size, seq_len, action_embedding_dim)

        # Concatenate shared features and action features along the last dimension
        critic_input_features = torch.cat([shared_features_seq, action_features_seq], dim=-1)
        # critic_input_features shape: (batch_size, seq_len, hidden_dim + action_embedding_dim)

        # Pass through critic heads
        q1_values_seq = self._critic1_head(critic_input_features) # Shape: (batch_size, seq_len, 1)
        q2_values_seq = self._critic2_head(critic_input_features) # Shape: (batch_size, seq_len, 1)

        return q1_values_seq, q2_values_seq, next_hidden_state

    def actor_parameters(self) -> Iterable[torch.nn.Parameter]:
        """ Return parameters of the actor components (shared net + actor head). """
        # Parameters from the shared backbone and the actor-specific head
        return list(self._shared_net.parameters()) + list(self._actor_head.parameters())

    def critic_parameters(self) -> Iterable[torch.nn.Parameter]:
        """ Return parameters of the critic components (shared net + action embed + critic heads). """
        # Note: Shared net params are included here too. Optimizers handle duplicates correctly.
        # Parameters from the shared backbone, action embedding layer, and both critic heads
        return list(self._shared_net.parameters()) + list(self._action_embed.parameters()) + \
               list(self._critic1_head.parameters()) + list(self._critic2_head.parameters())

