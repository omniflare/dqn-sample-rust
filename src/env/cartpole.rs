use rand::Rng;

#[derive(Debug, Clone, Copy, Default)]
pub struct CartPoleState {
    pub cart_position: f32,
    pub cart_velocity: f32,
    pub pole_angle: f32,
    pub pole_ang_velocity: f32,
}

#[derive(Clone, Copy)]
pub enum Action {
    Left = 0,
    Right = 1,
}

impl CartPoleState {
    pub fn to_array(&self) -> [f32; 4] {
        [
            self.cart_position,
            self.cart_velocity,
            self.pole_angle,
            self.pole_ang_velocity,
        ]
    }
}

impl Action {
    pub fn from_index(idx: usize) -> Self {
        if idx == 0 {
            Action::Left
        } else {
            Action::Right
        }
    }
}

pub struct CartPoleEnv {
    gravity: f32,
    pole_mass: f32,
    total_mass: f32,
    pole_half_length: f32,
    force_magnitude: f32,
    dt: f32,
    x_threshold: f32,
    theta_threshold: f32,
    state: CartPoleState,
    steps: u32,
    max_steps: u32,
    done: bool,
}

impl CartPoleEnv {
    pub fn new() -> Self {
        let cart_mass = 1.0;
        let pole_mass = 0.3;
        Self {
            gravity: 9.8,
            pole_mass,
            total_mass: cart_mass + pole_mass,
            pole_half_length: 0.5,
            force_magnitude: 10.0,
            dt: 0.02,
            x_threshold: 2.4,
            theta_threshold: 12.0 * std::f32::consts::PI / 180.0,
            state: CartPoleState::default(),
            steps: 0,
            max_steps: 500,
            done: false,
        }
    }

    pub fn reset(&mut self) -> CartPoleState {
        let mut rng = rand::thread_rng();
        self.state = CartPoleState {
            cart_position: rng.gen_range(-0.05..0.05),
            cart_velocity: rng.gen_range(-0.05..0.05),
            pole_angle: rng.gen_range(-0.05..0.05),
            pole_ang_velocity: rng.gen_range(-0.05..0.05),
        };
        self.steps = 0;
        self.done = false;
        self.state
    }

    pub fn step(&mut self, action: Action) -> (CartPoleState, f32, bool) {
        if self.done {
            //already ended episode, do nothing and return 0 reward;
            return (self.state, 0.0, true);
        }

        //    -F <-- [] --> + F
        let force = match action {
            Action::Left => -self.force_magnitude,
            Action::Right => self.force_magnitude,
        };

        let (x, x_dot, theta, theta_dot) = (
            self.state.cart_position,
            self.state.cart_velocity,
            self.state.pole_angle,
            self.state.pole_ang_velocity,
        );
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        // physics from lagrange
        let temp = (force + self.pole_mass * self.pole_half_length * theta_dot.powi(2) * sin_theta)
            / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp)
            / (self.pole_half_length
                * (4.0 / 3.0 - self.pole_mass * cos_theta.powi(2) / self.total_mass));
        let x_acc =
            temp - self.pole_mass * self.pole_half_length * theta_acc * cos_theta / self.total_mass;

        self.state = CartPoleState {
            cart_position: x + self.dt * x_dot,
            cart_velocity: x_dot + self.dt * x_acc,
            pole_angle: theta + self.dt * theta_dot,
            pole_ang_velocity: theta_dot + self.dt * theta_acc,
        };
        self.steps += 1;

        let terminated = self.state.cart_position.abs() > self.x_threshold
            || self.state.pole_angle.abs() > self.theta_threshold;
        self.done = terminated || self.steps >= self.max_steps;

        (self.state, if terminated { 0.0 } else { 1.0 }, self.done)
    }
    pub fn get_steps(&self) -> u32 { self.steps }
}
