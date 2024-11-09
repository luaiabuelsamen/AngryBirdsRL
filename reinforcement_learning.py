import tensorflow as tf
import tf_agents
from tf_agents.environments import gym_wrapper
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from main_game import AngryBirdsEnv
from tf_agents.environments import tf_py_environment  # To wrap the env in a TF environment

# Create the environment
env = AngryBirdsEnv()
env = gym_wrapper.GymWrapper(env)  # Wrap it in a GymWrapper for TF-Agents
env = tf_py_environment.TFPyEnvironment(env)  # Convert to a TF environment

# Check observation spec to ensure it's correct
print("Observation Spec:", env.observation_spec())  # Expect shape (12,)
print("Action Spec:", env.action_spec())  # Expect discrete action space

# Create a Q-network for DQN
q_net = q_network.QNetwork(
    env.observation_spec(),  # Observation shape (12,)
    env.action_spec(),
    fc_layer_params=(100, 50))  # Fully connected layers for the Q-Network

# Create the DQN agent
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

# Replay buffer to store experience
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=1,  # Collect data in batches of 1
    max_length=100000)

# Collect data from the environment and store it in the buffer
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()  # Get the full time_step from the environment

    # Add batch dimension to the observation if necessary
    if len(time_step.observation.shape) == 1:
        time_step = time_step._replace(observation=tf.expand_dims(time_step.observation, axis=0))

    action_step = policy.action(time_step)  # Pass the full time_step to the policy
    next_time_step = environment.step(action_step.action)  # Apply the action in the environment
    traj = trajectory.from_transition(time_step, action_step, next_time_step)  # Create a trajectory
    buffer.add_batch(traj)  # Store the trajectory in the replay buffer

# Train the agent
num_iterations = 10000
time_step = env.reset()  # Reset the environment and get the initial timestep

for _ in range(num_iterations):
    # Collect step
    collect_step(env, agent.collect_policy, replay_buffer)

    # Sample a batch of experience from the replay buffer
    experience, _ = replay_buffer.as_dataset(sample_batch_size=64, num_steps=2).take(1)

    # Train the agent with this batch
    train_loss = agent.train(experience)

    # Reset the environment if the current time_step is terminal
    if time_step.is_last():
        time_step = env.reset()

    if train_step_counter.numpy() % 100 == 0:
        print(f"Step {train_step_counter.numpy()}, Loss: {train_loss.loss}")

# After training, save the policy
from tf_agents.policies import policy_saver

# Directory to save the policy
policy_dir = './saved_policy'

# Save the agent's policy
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)

print(f"Policy saved to {policy_dir}")
