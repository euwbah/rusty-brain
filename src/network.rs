use layers::InputLayer;
use layers::OutputLayer;
use node::DerivativeCalculationParams;
use node::Node;

/// Default usage:
///
/// ```
/// let configs = NetworkConfigs {
///     learning_rate: 0.0002,
///     ..Default:default()
/// };
/// ```
pub struct NetworkConfigs {
    /// aka step size. See https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Background
    /// Default: 0.0001
    pub learning_rate: f64,
}

impl Default for NetworkConfigs {
    fn default() -> NetworkConfigs {
        NetworkConfigs {
            learning_rate: 0.0001,
        }
    }
}

/// Representing the entire neural network graph

pub struct Network {
    pub input_layer: InputLayer,
    pub output_layer: OutputLayer,
    pub network_configs: NetworkConfigs,
}

impl Network {
    pub fn new(input_layer: InputLayer, output_layer: OutputLayer) -> Network {
        Network {
            input_layer,
            output_layer,
            network_configs: Default::default(),
        }
    }

    pub fn set_network_configs(&mut self, network_configs: NetworkConfigs) {
        self.network_configs = network_configs;
    }

    /// Calculate the average loss on the entire training dataset
    pub fn calc_avg_training_loss(&self) {}

    /// Traverse through all the nodes in the network and evaluate d(loss) / d(node activation)
    /// for each one of them, storing them in the `TrainingState.dloss` field which can be
    /// retrieved with `Node.get_training_state()` or `Node.get_training_state_mut()`.
    ///
    /// `iteration`: The training iteration.
    /// `output_nodes_loss_fn_derivative`: Fn(Node name, node activation) -> derivative partial term
    pub fn evaluate_gradients(
        &mut self,
        iteration: i32,
        output_nodes_loss_fn_derivative: impl Fn(&str) -> f64 + 'static,
    ) {
        self.input_layer.set_iteration(iteration as usize);

        let derivative_calc_params = DerivativeCalculationParams::new(
            iteration,
            self.output_layer
                .output_nodes
                .iter()
                .map(|x| x.lock().unwrap().name().to_string())
                .collect(),
            output_nodes_loss_fn_derivative,
        );
        for n in &self.input_layer.input_nodes {
            let mut n = n.lock().unwrap();
            n.calc_activation_derivative(&derivative_calc_params);
        }
    }

    /// Update each node's weights based on its previously calculated gradients.
    /// Note that `evaluate_gradients()` must be called first.
    pub fn update_weights(&mut self) {}

    /// 1 epoch = go through all of the training data once.
    pub fn train_one_epoch(&mut self) {
        // Iteration represents the training sample index

        for iter in 0..self.input_layer.training_inputs.len() {
            self.input_layer.set_iteration(iter);
        }
    }
}
