from dataclasses import dataclass
import os

@dataclass
class Args:
    # Experiment specific arguments
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    """The name of the experiment"""
    seed: int = 1
    """The seed of the experiment"""
    torch_deterministic: bool = False
    """If toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """If toggled, cuda will be enabled if possible"""
    track: bool = False
    """If toggled, the experiment will be tracked with WandB"""
    wandb_project_name="rl2-darkroom-meta"
    """The WandB's project name"""
    wandb_entity: str = "king_arthur-org"
    """The entity of WandB's project"""
    capture_video: bool = False
    """If toggled, the video capture would be saved in the `videos` folder"""
    verbose: bool = False
    """If toggled, the episodic returns will be printed in the terminal when the environment is terminated"""

    # Algorithm specific arguments
    env_id: str = "Dark-Room-3x3-v0"
    """ID of the environment in Gymnasium"""
    env_specific_args: dict[str:any] | None = None
    """Environment specifc arguments"""
    total_timesteps: int = 10_000_000
    """The total amount of timesteps"""
    learning_rate: float  = 2.5e-4
    """The optimizer's learning rate"""
    adam_eps: float = 1e-5
    """The optmizer's eps"""
    num_envs: int = 8
    """The total amount of environments"""
    num_steps: int = 128
    """The number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """If toggled, the learning rate will decrease by annealing for policy and value networks"""
    gamma: float = 0.99
    """The discount factor value"""
    gae_lambda: float = 0.95
    """The strength of GAE"""
    num_minibatches: int = 4
    """The number of mini-batches"""
    update_epochs: int = 4
    """The number of epochs to update the policy"""
    norm_adv: bool = True
    """If toggled, the advantage will be normalized for a stable gradients"""
    clip_coef: float = 0.1
    """The surroggate clipping coefficient"""
    clip_vloss: bool = True
    """If toggled, a clipped loss for the value function will be used"""
    ent_coef: float = 0.01
    """The entropy coefficient value"""
    vf_coef: float = 0.5
    """The coefficient of the value function"""
    max_grad_norm: float = 0.5
    """The maximum norm for the gradient clipping"""
    target_kl: float = None
    """The target KL divergence threshold"""

    # Model specific arguments
    hidden_size: int = 128
    """The hidden_size that will be used in the internal layers"""
    num_layers: int = 1
    """The number of stacked LSTM layers"""

    # Must be filled in the runtime
    batch_size: int = 0
    """The batch size"""
    minibatch_size: int = 0
    """The mini-batch size"""
    num_iterations: int = 0
    """The number of iterations"""