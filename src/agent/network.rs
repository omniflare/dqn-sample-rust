use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct QNetwork<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    relu: Relu,
}

impl<B: Backend> QNetwork<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(4, 128).init(device),
            fc2: LinearConfig::new(128, 128).init(device),
            fc3: LinearConfig::new(128, 2).init(device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.fc1.forward(state));
        let x = self.relu.forward(self.fc2.forward(x));
        self.fc3.forward(x)
    }

    pub fn select_action(&self, state: &[f32; 4], device: &B::Device) -> usize {
        let state_tensor: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(state.as_slice(), device).unsqueeze_dim(0);
        let q_values: Vec<f32> = self
            .forward(state_tensor)
            .squeeze::<1>(0)
            .into_data()
            .to_vec()
            .unwrap();
        if q_values[0] > q_values[1] { 0 } else { 1 }
    }
}
