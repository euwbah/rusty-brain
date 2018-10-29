use layers::InputLayer;
use layers::OutputLayer;

/// Representing the entire neural network graph

pub struct Network {
    pub input_layer: InputLayer,
    pub output_layer: OutputLayer
}

impl Network {
    pub fn new(input_layer: InputLayer, output_layer: OutputLayer) -> Network {
        Network {
            input_layer,
            output_layer
        }
    }

    pub fn calc_avg_loss(&self) {

    }
}
