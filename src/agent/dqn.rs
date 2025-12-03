use crate::agent::ReplayBuffer;
use crate::agent::network::QNetwork;
use crate::agent::replay_buffer::Transition;

use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, backend::AutodiffBackend},
};
use rand::Rng;
use std::path::Path;

#[derive(Clone)]
pub struct DQNConfig {
    pub learning_rate: f64,
    pub gamma: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: f64,
    pub batch_size: usize,
    pub buffer_capacity: usize,
    pub target_update_freq: usize,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            batch_size: 64,
            buffer_capacity: 10000,
            target_update_freq: 100,
        }
    }
}

pub struct DQNAgent<B: AutodiffBackend> {
    q_network: QNetwork<B>,
    target_network: QNetwork<B::InnerBackend>,
    optimizer: OptimizerAdaptor<burn::optim::Adam, QNetwork<B>, B>,
    replay_buffer: ReplayBuffer,
    config: DQNConfig,
    epsilon: f64,
    steps: usize,
    device: B::Device,
}

impl<B: AutodiffBackend> DQNAgent<B>
where
    B::FloatElem: From<f32>,
    B::IntElem: From<i32>,
{
    pub fn new(config: DQNConfig, device: B::Device) -> Self {
        let q_network: QNetwork<B> = QNetwork::new(&device);
        let target_network = q_network.clone().valid();
        let optimizer = AdamConfig::new().init();
        let replay_buffer = ReplayBuffer::new(config.buffer_capacity);
        let epsilon = config.epsilon_start;

        Self {
            q_network,
            target_network,
            optimizer,
            replay_buffer,
            epsilon,
            steps: 0,
            device,
            config,
        }
    }
    pub fn select_action(&self, state: &[f32; 4]) -> usize {
        let mut rng = rand::thread_rng();
        if rng.r#gen::<f64>() < self.epsilon {
            rng.r#gen_range(0..2)
        } else {
            self.q_network.select_action(state, &self.device)
        }
    }

    pub fn select_action_greedy(&self, state: &[f32; 4]) -> usize {
        self.q_network.select_action(state, &self.device)
    }

    pub fn store_transition(&mut self, transition: Transition) {
        self.replay_buffer.push(transition);
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }

    pub fn get_epsilon(&self) -> f64 {
        self.epsilon
    }
    pub fn set_epsilon(&mut self, e: f64) {
        self.epsilon = e;
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.q_network
            .clone()
            .save_file(Path::new(path), &recorder)
            .map_err(|e| format!("Save failed: {:?}", e))?;
        Ok(())
    }

    pub fn train_step(&mut self) -> Option<f32> {
        if !self.replay_buffer.can_sample(self.config.batch_size) {
            return None;
        }
        let transitions = self.replay_buffer.sample(self.config.batch_size);
        let batch_size = transitions.len() as i32;

        let states: Vec<f32> = transitions.iter().flat_map(|t| t.state).collect();
        let states_tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(states.as_slice(), &self.device).reshape([batch_size, 4]);

        let actions: Vec<i32> = transitions.iter().map(|t| t.action as i32).collect();
        let actions_tensor: Tensor<B, 2, burn::tensor::Int> =
            Tensor::<B, 1, burn::tensor::Int>::from_ints(actions.as_slice(), &self.device)
                .reshape([batch_size, 1]);

        let rewards: Vec<f32> = transitions.iter().map(|t| t.reward).collect();
        let rewards_tensor: Tensor<B, 1> =
            Tensor::<B, 1>::from_floats(rewards.as_slice(), &self.device);

        let next_states: Vec<f32> = transitions.iter().flat_map(|t| t.next_state).collect();
        let next_states_tensor: Tensor<B::InnerBackend, 2> =
            Tensor::<B::InnerBackend, 1>::from_floats(next_states.as_slice(), &self.device)
                .reshape([batch_size, 4]);

        let dones: Vec<f32> = transitions
            .iter()
            .map(|t| if t.done { 1.0 } else { 0.0 })
            .collect();
        let dones_tensor: Tensor<B, 1> =
            Tensor::<B, 1>::from_floats(dones.as_slice(), &self.device);

        let current_q_all = self.q_network.forward(states_tensor);
        let current_q: Tensor<B, 1> = current_q_all.gather(1, actions_tensor).squeeze::<1>(1);

        let next_q_all = self.target_network.forward(next_states_tensor);
        let next_q_max: Tensor<B::InnerBackend, 1> = next_q_all.max_dim(1).squeeze(1);

        let gamma = Tensor::full(
            [batch_size as usize],
            self.config.gamma as f32,
            &self.device,
        );
        let not_done: Tensor<B, 1> =
            Tensor::ones([batch_size as usize], &self.device) - dones_tensor;
        let next_q_max_ad: Tensor<B, 1> = Tensor::from_inner(next_q_max);
        let target_q = rewards_tensor + gamma * next_q_max_ad * not_done;

        let diff = current_q - target_q;
        let loss = (diff.clone() * diff).mean();

        let grads = GradientsParams::from_grads(loss.backward(), &self.q_network);
        self.q_network =
            self.optimizer
                .step(self.config.learning_rate, self.q_network.clone(), grads);

        self.steps += 1;
        if self.steps % self.config.target_update_freq == 0 {
            self.target_network = self.q_network.clone().valid();
        }

        Some(loss.into_data().to_vec().unwrap()[0])
    }

    pub fn load(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.q_network = self
            .q_network
            .clone()
            .load_file(Path::new(path), &recorder, &self.device)
            .map_err(|e| format!("Load failed: {:?}", e))?;
        self.target_network = self.q_network.clone().valid();
        Ok(())
    }
}
