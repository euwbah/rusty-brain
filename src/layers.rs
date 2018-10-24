use node::Node;
use AM;
use ndarray::prelude::Array2;
use ndarray::Axis;
use ndarray::ArrayBase;
use node::InputNode;

pub struct InputLayer {
    pub input_nodes: Vec<AM<InputNode>>,
    pub training_inputs: Array2<f64>,
}

impl InputLayer {
    /// `training_vals` is a flattened array of training input values
    /// `training_vals[0 .. nodes.len()]` represents one single input value
    /// where each of the values corresponds to one input node, according to the same index.
    pub fn new(nodes: &Vec<AM<InputNode>>, training_vals: &[f64]) -> InputLayer {
        let mut input_nodes = vec![];
        for node in nodes {
            input_nodes.push(node.clone())
        }

        let node_count = input_nodes.len();

        assert!(training_vals.len() % node_count == 0, "training_vals.len() must be a multiple of nodes.len()!");

        let mut training_inputs =
            Array2::<f64>::zeros((training_vals.len() / node_count, node_count));

        for (idx, val) in training_vals.iter().enumerate() {
            let row = idx / node_count;
            let column = idx % node_count;

            training_inputs[[row, column]] = *val
        }

        InputLayer {
            input_nodes,
            training_inputs,
        }
    }

    pub fn set_iteration(&mut self, iter: usize) {
        println!("Setting input iteration: {}", iter);

        let vals = self.training_inputs.slice(s![iter, ..]);
        for (idx, val) in vals.iter().enumerate() {
            let mut node = self.input_nodes[idx].lock().unwrap();

            println!("Assigning input [{}] to {}", node.name(), val);
            node.value = *val;
        }
    }
}

pub struct OutputLayer {
    pub output_nodes: Vec<AM<Node>>,
    pub training_ground_truths: Array2<f64>,
    pub loss_function: Box<Fn(&Vec<AM<Node>>, &[f64]) -> f64>,
}

impl OutputLayer {
    /// `training_vals` is a flattened array of training ground truth values
    /// `training_vals[0 .. nodes.len()]` represents one single ground truth value
    /// where each of the values corresponds to the output nodes, according to the same index.
    pub fn new(nodes: &Vec<AM<Node>>,
               training_vals: &[f64],
               loss_function: Box<Fn(&Vec<AM<Node>>, &[f64]) -> f64>) -> OutputLayer {
        let mut output_nodes = vec![];
        for node in nodes {
            output_nodes.push(node.clone())
        }

        let node_count = output_nodes.len();

        assert!(training_vals.len() % node_count == 0, "training_vals.len() must be a multiple of nodes.len()!");

        let mut training_ground_truths =
            Array2::<f64>::zeros((training_vals.len() / node_count, node_count));

        for (idx, val) in training_vals.iter().enumerate() {
            let row = idx / node_count;
            let column = idx % node_count;

            training_ground_truths[[row, column]] = *val;
        }

        OutputLayer {
            output_nodes,
            training_ground_truths,
            loss_function
        }
    }

    /// Calculates the loss of one particular iteration
    pub fn calculate_iter_loss(&self, iter: usize) {
        let vals = self.training_ground_truths.slice(s![iter, ..]);
        for (idx, node_ground_truth_val) in vals.iter().enumerate() {
            let mut node = self.output_nodes[idx].lock().unwrap();

            let activation = node.calc_activation();
            println!("activation: {}, ground: {}", activation, node_ground_truth_val);
        }
    }
}
