# Angry Birds RL Agent

This project uses Reinforcement Learning (RL) to train an agent to play the Angry Birds game. The agent is trained using the PPO (Proximal Policy Optimization) algorithm from Stable Baselines 3. It logs training progress and performance metrics using WandB and provides evaluation functionality.

## Requirements

To set up and run this project, make sure you have the following dependencies installed:

- Python 3.10 or higher
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
- [WandB](https://wandb.ai/)
- [PyTorch](https://pytorch.org/)
- [TensorFlow (optional)](https://www.tensorflow.org/)
- [tqdm and rich](https://github.com/tqdm/tqdm) (for progress bar functionality)

### Install dependencies

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

Or, for extra functionality (including the progress bar), use:

```bash
pip install stable-baselines3[extra]
```

### Create a `requirements.txt` for convenience

```txt
stable-baselines3
wandb
torch
tqdm
rich
```

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/angry-birds-rl.git
   cd angry-birds-rl
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up WandB (if not already set up):**

   If you don’t have a WandB account, you can create one [here](https://wandb.ai/). Then, log in:

   ```bash
   wandb login
   ```

4. **Ensure your environment supports GPU (optional):**

   If you have a CUDA-capable GPU, make sure you have the correct drivers installed and that PyTorch is configured to use CUDA. Otherwise, the script will default to CPU.

## Training the Agent

To start training the agent, run the following command:

```bash
python train_agent.py
```

### Arguments:

- `game`: Instance of the `Game` class used in the training.
- `total_timesteps`: Total training steps (default is 1,000,000).
- `eval_freq`: Frequency of evaluation in terms of timesteps (default is 10,000).
- `save_freq`: Frequency of saving model checkpoints in terms of timesteps (default is 50,000).
- `log_freq`: Frequency of logging metrics (default is 1,000).
- `n_envs`: Number of parallel environments for training (default is 4).
- `load_path`: Path to load a pre-trained model (optional).

Example:

```bash
python train_agent.py
```

This will train the agent on the Angry Birds game using the PPO algorithm, logging training metrics to WandB and saving periodic checkpoints.

## Evaluating the Agent

Once the agent is trained, you can evaluate its performance using the following command:

```bash
python evaluate_agent.py
```

The agent will be evaluated across 100 episodes, and the results (mean reward, success rate, etc.) will be printed to the console.

### Evaluation Metrics:

- **Mean Reward**: Average reward per episode.
- **Success Rate**: Percentage of episodes where the agent succeeds.
- **Mean Episode Length**: Average number of steps per episode.

## Logging and Checkpoints

- **WandB**: Logs training metrics to [WandB](https://wandb.ai/), including rewards and training time.
- **Checkpoints**: Models are saved every `save_freq` timesteps.
- **Training Stats**: Stats are saved periodically and include the best mean reward and the total training time.

## Example Output

When training completes, the following files will be saved:

- `final_model`: The final trained model.
- `final_stats.json`: Training stats including the final mean reward and total timesteps.
- `checkpoints/`: Folder containing checkpointed models.
- `logs/`: Folder containing training logs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) for RL training.
- [WandB](https://wandb.ai/) is used for logging and visualization.
- Thanks to the creators of the Angry Birds game and for their inspiration in creating this project!