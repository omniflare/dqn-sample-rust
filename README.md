# CartPole DQN in Rust 🦀🤖

A Deep Q-Network (DQN) reinforcement learning agent that learns to balance a pole on a cart, implemented entirely in Rust with real-time visualization.

![CartPole Animation](https://user-images.githubusercontent.com/placeholder/cartpole.gif)

## 🎯 What is This?

This project implements the classic CartPole environment and trains a DQN agent to solve it. The agent learns through trial and error to balance a pole on a moving cart by deciding whether to push left or right. The application features a beautiful real-time GUI that visualizes the training process as it happens.

### Key Features

- **Pure Rust Implementation**: Built with modern Rust, leveraging the Burn deep learning framework
- **Real-time Visualization**: Watch the agent learn in real-time with an interactive GUI powered by egui/eframe
- **GPU Acceleration**: Uses WGPU backend for fast neural network training
- **Experience Replay**: Implements a replay buffer for stable learning
- **Model Persistence**: Save and load trained models for continued training or inference
- **Live Training Metrics**: View episode rewards, loss curves, epsilon decay, and performance statistics

## 🏗️ Architecture

### Components

1. **DQN Agent** (`src/agent/`)
   - Neural network with 3 fully connected layers (4 → 128 → 128 → 2)
   - Experience replay buffer with configurable capacity
   - Epsilon-greedy exploration strategy with decay
   - Separate target network for stable Q-value estimation
   - Adam optimizer with configurable learning rate

2. **CartPole Environment** (`src/env/`)
   - Physics-based simulation using Lagrangian mechanics
   - Configurable parameters (gravity, pole mass, force magnitude)
   - State space: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
   - Action space: {Left, Right}
   - Episode termination conditions: pole angle > 12°, cart position > 2.4, or 500 steps

3. **UI** (`src/ui/`)
   - Real-time CartPole rendering with smooth animations
   - Live reward history graph (last 100 episodes)
   - Training controls (Start/Stop)
   - Statistics dashboard (episode count, steps, epsilon, loss)
   - Model save/load status indicators

### Tech Stack

- **[Burn](https://github.com/burn-rs/burn)**: Deep learning framework with autodifferentiation
- **[egui/eframe](https://github.com/emilk/egui)**: Immediate mode GUI framework
- **[WGPU](https://wgpu.rs/)**: GPU acceleration backend
- **[Tokio](https://tokio.rs/)**: Async runtime for training loop
- **[Flume](https://github.com/zesterer/flume)**: MPSC channels for thread communication

## 🚀 Getting Started

### Prerequisites

- Rust 1.70+ (2024 edition)
- GPU with Vulkan/Metal/DirectX 12 support (for WGPU backend)

### Installation

```bash
git clone https://github.com/omniflare/dqn-sample-rust.git
cd dqn-sample-rust
cargo build --release
```

### Running

```bash
cargo run --release
```

The application will:
1. Launch a GUI window showing the CartPole environment
2. Attempt to load a previously saved model (if available)
3. Wait for you to press "Start Training" or run inference with the loaded model

### Usage

**Training Mode:**
1. Click "Start Training" to begin the learning process
2. Watch as the agent explores and learns to balance the pole
3. Monitor the reward graph to track learning progress
4. Click "Stop Training" to pause and save the model

**Inference Mode:**
- With a trained model loaded, the agent will continuously demonstrate its learned policy
- The epsilon is automatically set to 0.05 for near-greedy behavior

## 📊 Hyperparameters

Default configuration (in `DQNConfig`):

```rust
learning_rate: 0.001       // Adam optimizer learning rate
gamma: 0.99                // Discount factor for future rewards
epsilon_start: 1.0         // Initial exploration rate
epsilon_end: 0.01          // Minimum exploration rate
epsilon_decay: 0.995       // Decay rate per episode
batch_size: 64             // Minibatch size for training
buffer_capacity: 10000     // Replay buffer size
target_update_freq: 100    // Steps between target network updates
```

## 🎓 How It Works

### DQN Algorithm

1. **Experience Collection**: Agent interacts with environment, storing (state, action, reward, next_state, done) transitions
2. **Replay Sampling**: Random minibatch sampled from replay buffer
3. **Q-Value Estimation**: 
   - Current Q: Q(s, a) from policy network
   - Target Q: r + γ * max_a' Q'(s', a') from target network
4. **Loss Calculation**: MSE between current Q and target Q
5. **Network Update**: Backpropagation and gradient descent
6. **Target Network Sync**: Periodically copy policy network to target network

### CartPole Physics

The environment simulates a pole attached to a cart using Lagrangian mechanics:

- **State**: [x, ẋ, θ, θ̇] (position, velocity, angle, angular velocity)
- **Dynamics**: Second-order differential equations solved with Euler integration
- **Reward**: +1 for each timestep the pole remains balanced, 0 on failure
- **Episode End**: |x| > 2.4, |θ| > 12°, or 500 steps reached

## 📈 Expected Results

A well-trained agent should:
- Consistently achieve 200-500 steps per episode
- Maintain the pole within ±5° of vertical
- Show smooth, minimal cart movements

Training typically converges within:
- **~100-200 episodes** for basic balancing
- **~300-500 episodes** for optimal performance

## 🔧 Customization

### Modify Network Architecture

Edit `src/agent/network.rs`:

```rust
fc1: LinearConfig::new(4, 256).init(device),  // Increase hidden units
fc2: LinearConfig::new(256, 256).init(device),
fc3: LinearConfig::new(256, 2).init(device),
```

### Adjust Training Speed

In `src/main.rs`, modify sleep durations:

```rust
tokio::time::sleep(Duration::from_micros(100)).await;  // Training speed
tokio::time::sleep(Duration::from_millis(20)).await;   // Inference speed
```

### Change Environment Parameters

Edit `CartPoleEnv::new()` in `src/env/cartpole.rs`:

```rust
force_magnitude: 15.0,    // Stronger pushes
max_steps: 1000,          // Longer episodes
theta_threshold: 15.0 * PI / 180.0,  // More lenient angle
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add prioritized experience replay
- [ ] Implement More Models.
- [ ] Add more environments (MountainCar, Acrobot, etc.)
- [ ] Hyperparameter tuning UI
- [ ] Better reward visualization (moving average, confidence intervals)
- [ ] Export training videos/GIFs

## 📚 Resources

- [DQN Paper](https://arxiv.org/abs/1312.5602) - Original Deep Q-Learning paper
- [Burn Framework](https://burn.dev/) - Rust deep learning framework documentation
- [OpenAI Gym CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) - Reference implementation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Burn team for the excellent deep learning framework
- egui community for the fantastic immediate mode GUI library
- OpenAI Gym for the CartPole environment specification
- My College course that introduced me to this subject
---

**Built with ❤️ and 🦀 by [omniflare](https://github.com/omniflare)**
