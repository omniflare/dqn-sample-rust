use eframe::egui;
use flume::{Receiver, Sender};
use std::f32::consts::PI;

use crate::env::CartPoleState;

#[derive(Debug, Clone)]
pub enum UICommand {
    Start,
    Stop,
}

#[derive(Debug, Clone)]
pub enum TrainingUpdate {
    State(CartPoleState, bool, u32),
    Stats {
        episode: usize,
        steps: usize,
        reward: f32,
        epsilon: f64,
        loss: f32,
    },
    TrainingStarted,
    TrainingStopped,
    ModelSaved,
    ModelLoaded,
}

pub struct CartPoleApp {
    cmd_tx: Sender<UICommand>,
    update_rx: Receiver<TrainingUpdate>,
    state: CartPoleState,
    is_done: bool,
    step_count: u32,
    is_training: bool,
    episode: usize,
    total_steps: usize,
    reward: f32,
    epsilon: f64,
    loss: f32,
    rewards: Vec<f32>,
    status_msg: Option<String>,
}

impl CartPoleApp {
    pub fn new(
        _cc: &eframe::CreationContext<'_>,
        cmd_tx: Sender<UICommand>,
        update_rx: Receiver<TrainingUpdate>,
    ) -> Self {
        Self {
            cmd_tx,
            update_rx,
            state: CartPoleState::default(),
            is_done: false,
            step_count: 0,
            is_training: false,
            episode: 0,
            total_steps: 0,
            reward: 0.0,
            epsilon: 1.0,
            loss: 0.0,
            rewards: Vec::new(),
            status_msg: None,
        }
    }

    fn process_updates(&mut self) {
        while let Ok(update) = self.update_rx.try_recv() {
            match update {
                TrainingUpdate::State(s, done, steps) => {
                    self.state = s;
                    self.is_done = done;
                    self.step_count = steps;
                }
                TrainingUpdate::Stats {
                    episode,
                    steps,
                    reward,
                    epsilon,
                    loss,
                } => {
                    self.episode = episode;
                    self.total_steps = steps;
                    self.reward = reward;
                    self.epsilon = epsilon;
                    self.loss = loss;
                    self.rewards.push(reward);
                    if self.rewards.len() > 100 {
                        self.rewards.remove(0);
                    }
                }
                TrainingUpdate::TrainingStarted => {
                    self.is_training = true;
                    self.status_msg = Some("Training started".into());
                }
                TrainingUpdate::TrainingStopped => {
                    self.is_training = false;
                    self.status_msg = Some("Training stopped".into());
                }
                TrainingUpdate::ModelSaved => self.status_msg = Some("Model saved".into()),
                TrainingUpdate::ModelLoaded => self.status_msg = Some("Model loaded".into()),
            }
        }
    }

    fn render_cartpole(&self, ui: &mut egui::Ui) {
        let size = ui.available_size();
        let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
        let rect = resp.rect;

        let bg = if self.is_done {
            egui::Color32::from_rgb(50, 20, 20)
        } else {
            egui::Color32::from_rgb(20, 20, 40)
        };
        painter.rect_filled(rect, 0.0, bg);

        let cx = rect.center().x;
        let ground_y = rect.min.y + rect.height() * 0.75;
        let scale = 100.0;

        let cart_x = cx + self.state.cart_position * scale;
        let cart_y = ground_y - 30.0;
        let cart_rect =
            egui::Rect::from_center_size(egui::pos2(cart_x, cart_y), egui::vec2(80.0, 40.0));
        painter.rect_filled(cart_rect, 4.0, egui::Color32::from_rgb(79, 195, 247));

        painter.circle_filled(
            egui::pos2(cart_x - 25.0, cart_y + 20.0),
            10.0,
            egui::Color32::DARK_GRAY,
        );
        painter.circle_filled(
            egui::pos2(cart_x + 25.0, cart_y + 20.0),
            10.0,
            egui::Color32::DARK_GRAY,
        );

        painter.line_segment(
            [
                egui::pos2(rect.min.x + 50.0, ground_y),
                egui::pos2(rect.max.x - 50.0, ground_y),
            ],
            egui::Stroke::new(3.0, egui::Color32::GRAY),
        );

        let pole_len = 150.0;
        let pole_start = egui::pos2(cart_x, cart_y - 20.0);
        let pole_end = egui::pos2(
            cart_x + pole_len * self.state.pole_angle.sin(),
            cart_y - 20.0 - pole_len * self.state.pole_angle.cos(),
        );
        let angle_ratio = (self.state.pole_angle.abs() / (12.0 * PI / 180.0)).min(1.0);
        let pole_color = egui::Color32::from_rgb(
            (100.0 + 155.0 * angle_ratio) as u8,
            (200.0 * (1.0 - angle_ratio)) as u8,
            50,
        );
        painter.line_segment([pole_start, pole_end], egui::Stroke::new(8.0, pole_color));
        painter.circle_filled(pole_start, 8.0, egui::Color32::GOLD);
        painter.circle_filled(pole_end, 6.0, egui::Color32::from_rgb(255, 87, 34));

        let info = format!(
            "Step: {}  |  Pos: {:.2}  |  Angle: {:.1}°",
            self.step_count,
            self.state.cart_position,
            self.state.pole_angle.to_degrees()
        );
        painter.text(
            rect.min + egui::vec2(10.0, 10.0),
            egui::Align2::LEFT_TOP,
            info,
            egui::FontId::proportional(14.0),
            egui::Color32::WHITE,
        );
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("DQN Training");
        ui.add_space(10.0);

        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.is_training, egui::Button::new("▶ Train"))
                .clicked()
            {
                let _ = self.cmd_tx.send(UICommand::Start);
            }
            if ui
                .add_enabled(self.is_training, egui::Button::new("⏹ Stop"))
                .clicked()
            {
                let _ = self.cmd_tx.send(UICommand::Stop);
            }
        });

        let status = if self.is_training {
            ("● Training", egui::Color32::GREEN)
        } else {
            ("○ Idle", egui::Color32::YELLOW)
        };
        ui.colored_label(status.1, status.0);

        if let Some(msg) = &self.status_msg {
            ui.label(msg);
        }

        ui.separator();
        ui.label("Statistics");

        egui::Grid::new("stats").show(ui, |ui| {
            ui.label("Episode:");
            ui.label(format!("{}", self.episode));
            ui.end_row();
            ui.label("Steps:");
            ui.label(format!("{}", self.total_steps));
            ui.end_row();
            ui.label("Reward:");
            ui.label(format!("{:.0}", self.reward));
            ui.end_row();
            ui.label("Avg (100):");
            let avg = if self.rewards.is_empty() {
                0.0
            } else {
                self.rewards.iter().sum::<f32>() / self.rewards.len() as f32
            };
            ui.label(format!("{:.1}", avg));
            ui.end_row();
            ui.label("Epsilon:");
            ui.label(format!("{:.3}", self.epsilon));
            ui.end_row();
            ui.label("Loss:");
            ui.label(format!("{:.5}", self.loss));
            ui.end_row();
        });

        ui.separator();
        ui.label("Reward History");
        self.render_chart(ui);
    }

    fn render_chart(&self, ui: &mut egui::Ui) {
        let (resp, painter) =
            ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
        let rect = resp.rect;
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(30, 30, 50));

        if self.rewards.is_empty() {
            return;
        }

        let max_r = self.rewards.iter().cloned().fold(1.0f32, f32::max);
        let n = self.rewards.len().min(50);
        let bar_w = (rect.width() - 10.0) / n as f32;
        let data: Vec<_> = self.rewards.iter().rev().take(n).rev().cloned().collect();

        for (i, r) in data.iter().enumerate() {
            let h = (r / max_r * 50.0).max(2.0);
            let x = rect.min.x + 5.0 + i as f32 * bar_w;
            let color = if *r > max_r * 0.7 {
                egui::Color32::GREEN
            } else {
                egui::Color32::YELLOW
            };
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(x, rect.max.y - 5.0 - h),
                    egui::vec2(bar_w - 1.0, h),
                ),
                1.0,
                color,
            );
        }
    }
}

impl eframe::App for CartPoleApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_updates();
        ctx.request_repaint();

        egui::SidePanel::right("controls")
            .min_width(220.0)
            .show(ctx, |ui| {
                ui.add_space(10.0);
                self.render_controls(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| self.render_cartpole(ui));
    }
}
