mod agent;
mod env;
mod ui;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use eframe::egui;
use flume::{Receiver, Sender};

use crate::agent::{DQNAgent, DQNConfig, Transition};
use crate::env::{Action, CartPoleEnv};
use crate::ui::{CartPoleApp, TrainingUpdate, UICommand};

type Backend = Autodiff<Wgpu>;
const MODEL_PATH: &str = "cartpole_model";

async fn training_loop(
    cmd_rx: Receiver<UICommand>,
    update_tx: Sender<TrainingUpdate>,
    stop: Arc<AtomicBool>,
) {
    let device = WgpuDevice::default();
    let mut agent: DQNAgent<Backend> = DQNAgent::new(DQNConfig::default(), device);
    let mut env = CartPoleEnv::new();

    if agent.load(MODEL_PATH).is_ok() {
        let _ = update_tx.send(TrainingUpdate::ModelLoaded);
        agent.set_epsilon(0.05);
    }

    let mut is_training = false;
    let mut episode = 0;
    let mut total_steps: usize = 0;
    let mut state = env.reset();

    let _ = update_tx.send(TrainingUpdate::State(state, false, 0));

    while !stop.load(Ordering::Relaxed) {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                UICommand::Start => {
                    is_training = true;
                    let _ = update_tx.send(TrainingUpdate::TrainingStarted);
                }
                UICommand::Stop => {
                    is_training = false;
                    let _ = agent.save(MODEL_PATH);
                    let _ = update_tx.send(TrainingUpdate::ModelSaved);
                    let _ = update_tx.send(TrainingUpdate::TrainingStopped);
                }
            }
        }

        let action_idx = if is_training {
            agent.select_action(&state.to_array())
        } else {
            agent.select_action_greedy(&state.to_array())
        };

        let (next_state, reward, done) = env.step(Action::from_index(action_idx));
        total_steps += 1;

        if is_training {
            agent.store_transition(Transition::new(
                state.to_array(),
                action_idx,
                reward,
                next_state.to_array(),
                done,
            ));
            let loss = agent.train_step().unwrap_or(0.0);
            let _ = update_tx.send(TrainingUpdate::State(next_state, done, env.get_steps()));

            if done {
                agent.decay_epsilon();
                let _ = update_tx.send(TrainingUpdate::Stats {
                    episode,
                    steps: total_steps,
                    reward: env.get_steps() as f32,
                    epsilon: agent.get_epsilon(),
                    loss,
                });
                episode += 1;
                state = env.reset();
            } else {
                state = next_state;
            }
            tokio::time::sleep(Duration::from_micros(100)).await;
        } else {
            let _ = update_tx.send(TrainingUpdate::State(next_state, done, env.get_steps()));
            if done {
                episode += 1;
                state = env.reset();
            } else {
                state = next_state;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    if is_training {
        let _ = agent.save(MODEL_PATH);
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let (cmd_tx, cmd_rx) = flume::bounded::<UICommand>(100);
    let (update_tx, update_rx) = flume::bounded::<TrainingUpdate>(1000);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    let handle = std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
            .block_on(training_loop(cmd_rx, update_tx, stop_clone));
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 600.0])
            .with_title("CartPole DQN"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    let _ = eframe::run_native(
        "CartPole DQN",
        options,
        Box::new(move |cc| Ok(Box::new(CartPoleApp::new(cc, cmd_tx, update_rx)))),
    );

    stop.store(true, Ordering::Relaxed);
    let _ = handle.join();
}
