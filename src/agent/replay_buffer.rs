use rand::seq::SliceRandom;

#[derive(Clone)]
pub struct Transition {
    pub state: [f32; 4],
    pub action: usize,
    pub reward: f32,
    pub next_state: [f32; 4],
    pub done: bool,
}

impl Transition {
    pub fn new(state: [f32; 4], action: usize, reward: f32, next_state: [f32; 4], done: bool) -> Self {
        Self { state, action, reward, next_state, done }
    }
}

pub struct ReplayBuffer {
    buffer: Vec<Transition>,
    capacity: usize,
    position: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { buffer: Vec::with_capacity(capacity), capacity, position: 0 }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Transition> {
        let mut rng = rand::thread_rng();
        self.buffer.choose_multiple(&mut rng, batch_size.min(self.buffer.len())).cloned().collect()
    }

    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.buffer.len() >= batch_size
    }
}
