from abc import abstractmethod
from typing import Tuple

import torch

from drl.net import RecurrentNetwork
from drl.policy_dist import CategoricalDist

class PretrainedRecurrentNetwork(RecurrentNetwork):
    """
    Pretrained recurrent network.
    """
    
    @abstractmethod
    def forward(
        self, 
        obs_seq: torch.Tensor, 
        hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution using the recurrent network.
        
        It's recommended to set your recurrent network to `batch_first=True`.

        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            policy_dist_seq (CategoricalDist): policy distribution sequences
            next_seq_hidden_state (Tensor): hidden state which will be used for the next sequence
            
        ## Input/Output Details
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(seq_batch_size, seq_len, *obs_shape)`|
        |hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy_dist_seq|`*batch_shape` = `(seq_batch_size, seq_len)`, details in `PolicyDist` docs|
        |next_seq_hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Refer to the following explanation:
        
        * `seq_batch_size`: the size of sequence batch
        * `seq_len`: the length of each sequence
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network
        
        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError

class RecurrentPPONetwork(RecurrentNetwork):
    """
    Recurrent Proximal Policy Optimization (PPO) shared network.
    
    Since it uses the recurrent network, 
    you must consider the hidden state which can acheive the action-observation history.
    
    Note that since it uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    """
        
    @abstractmethod
    def forward(
        self, 
        obs_seq: torch.Tensor, 
        hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution 
        and state value using the recurrent network.
        
        It's recommended to set your recurrent network to `batch_first=True`.

        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            policy_dist_seq (CategoricalDist): policy distribution sequences
            state_value_seq (Tensor): state value sequences
            next_seq_hidden_state (Tensor): hidden state which will be used for the next sequence
            
        ## Input/Output Details
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(seq_batch_size, seq_len, *obs_shape)`|
        |hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy_dist_seq|`*batch_shape` = `(seq_batch_size, seq_len)`, details in `PolicyDist` docs|
        |state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |next_seq_hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Refer to the following explanation:
        
        * `seq_batch_size`: the size of sequence batch
        * `seq_len`: the length of each sequence
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network
        
        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError
    
class RecurrentPPORNDNetwork(RecurrentNetwork):
    """
    Recurrent Proximal Policy Optimization (PPO) shared network with Random Network Distillation (RND).
    
    Since it uses the recurrent network, you must consider the hidden state which can acheive the action-observation history.
    
    Note that since PPO uses the Actor-Critic architecure and the parameter sharing, 
    the encoding layer must be shared between Actor and Critic. 
    Be careful not to share parameters between PPO and RND networks.
    
    RND uses episodic and non-episodic reward streams. 
    RND constitutes of the predictor and target networks. 
    Both of them should have the similar architectures (not must same) but their initial parameters should not be the same.
    The target network is determinsitic, which means it will be never updated. 
    """
        
    @abstractmethod
    def forward_actor_critic(
        self, 
        obs_seq: torch.Tensor, 
        hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method to compute policy distribution, 
        episodic state value and non-episodic state value using the recurrent network.
                
        It's recommended to set your recurrent network to `batch_first=True`.
        
        Args:
            obs_seq (Tensor): observation sequences
            hidden_state (Tensor): hidden states at the beginning of each sequence

        Returns:
            policy_dist_seq (Tensor): policy distribution sequences
            epi_state_value_seq (Tensor): episodic state value sequences
            nonepi_state_value_seq (Tensor): non-episodic state value sequences
            next_seq_hidden_state (Tensor): hidden state which will be used for the next sequence
        
        ## Input/Output Details
                
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs_seq|`(seq_batch_size, seq_len, *obs_shape)`|
        |hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Output:
        
        |Output|Shape|
        |:---|:---|
        |policy_dist_seq|`*batch_shape` = `(seq_batch_size, seq_len)`, details in `PolicyDist` docs|
        |epi_state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |nonepi_state_value_seq|`(seq_batch_size, seq_len, 1)`|
        |next_seq_hidden_state|`(D x num_layers, seq_batch_size, H)`|
        
        Refer to the following explanation:
        
        * `seq_batch_size`: the size of sequence batch
        * `seq_len`: the length of each sequence
        * `num_layers`: the number of recurrent layers
        * `D`: 2 if bidirectional otherwise 1
        * `H`: the value depends on the type of the recurrent network
        
        When you use LSTM, `H` = `H_cell` + `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html. 
        When you use GRU, `H` = `H_out`. See details in https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_rnd(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ## Summary
        
        Feed forward method tp to compute both predicted feature and target feature. 
        You can use the hidden state by concatenating it with either the observation or the embedding of the observation. 

        Args:
            obs (Tensor): observation batch
            hidden_state (Tensor): hidden state batch with flattened features

        Returns:
            predicted_feature (Tensor): predicted feature whose gradient flows
            target_feature (Tensor): target feature whose gradient doesn't flow
            
        ## Input/Output Details
        
        The value of `out_features` depends on you.
        
        Input:
        
        |Input|Shape|
        |:---|:---|
        |obs|`(batch_size, *obs_shape)`|
        |hidden_state|`(batch_size, D x num_layers x H)`|
        
        Output:
        
        |Input|Shape|
        |:---|:---|
        |predicted_feature|`(batch_size, out_features)`|
        |target_feature|`(batch_size, out_features)`|
        """
        raise NotImplementedError

from abc import abstractmethod
from typing import Tuple, Union, Iterable

import torch
import torch.nn as nn

from drl.net import RecurrentNetwork, Network # Assuming drl.net contains base Network/RecurrentNetwork
from drl.policy_dist import CategoricalDist # Keep for PPO/Pretrained


# --- New Interface for Recurrent SAC Agent ---
class RecurrentSACNetwork(RecurrentNetwork):
    """
    Recurrent Soft Actor-Critic (SAC) network interface.
    Requires separate actor and critic components.
    """

    @abstractmethod
    def forward_actor(
        self,
        obs_seq: torch.Tensor,
        hidden_state: torch.Tensor
    ) -> Tuple[CategoricalDist, torch.Tensor]:
        """
        Forward pass for the actor network.

        Args:
            obs_seq (torch.Tensor): Observation sequences (batch_size, seq_len, *obs_shape).
            hidden_state (torch.Tensor): Actor's hidden states (num_layers * D, batch_size, H_actor).

        Returns:
            Tuple[CategoricalDist, torch.Tensor]:
                - Policy distribution sequences.
                - Next actor hidden state (num_layers * D, batch_size, H_actor).
        """
        raise NotImplementedError

    @abstractmethod
    def forward_critic(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor, # Action sequences needed for Q-value estimation
        hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the critic network(s). SAC typically uses two Q-networks.

        Args:
            obs_seq (torch.Tensor): Observation sequences (batch_size, seq_len, *obs_shape).
            action_seq (torch.Tensor): Action sequences (batch_size, seq_len, *action_shape).
            hidden_state (torch.Tensor): Critic's hidden states (num_layers * D, batch_size, H_critic).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Q1 value sequences (batch_size, seq_len, 1).
                - Q2 value sequences (batch_size, seq_len, 1).
                - Next critic hidden state (num_layers * D, batch_size, H_critic).
        """
        raise NotImplementedError

    @abstractmethod
    def actor_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters of the actor network."""
        raise NotImplementedError

    @abstractmethod
    def critic_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters of the critic network(s)."""
        raise NotImplementedError
