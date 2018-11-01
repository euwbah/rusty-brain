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

    /// Assigns the input_nodes input values based on the iteration.
    ///
    /// The `iter` parameter is a ring which wraps around 0 and `self.training_inputs.len()`
    /// E.g. assuming iter is 5, and `training_inputs` has 3 vectors of node input values,
    /// the nodes will be assigned to the values given by index 2 (5 % 3) of `training_inputs`.
    pub fn set_iteration(&mut self, iter: usize) {
        println!("Setting input iteration: {}", iter);

        let idx = iter % self.training_inputs.len();

        let vals = self.training_inputs.slice(s![idx, ..]);
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
    pub loss_function: Box<Fn(Vec<f64>, Vec<f64>) -> f64>,
}

impl OutputLayer {
    /// `_training_ground_truths` is a flattened array of training ground truth values
    /// `_training_ground_truths[0 .. nodes.len()]` represents one single ground truth value
    /// where each of the values corresponds to the output nodes, according to the same index.
    ///
    /// `loss_function` is a function that accepts two params:
    /// `| output_node_activations, expected_ground_truths |`.
    /// `output_node_activations` is a `Vec<f64>` that lists the activation values of the output nodes
    /// in the same index order as `self.output_nodes`.
    /// `expected_ground_truths` represents the correct activation values of each output node
    /// for any particular iteration in the same index order as `self.output_nodes`.
    /// The `loss_function` should calculate and return the loss score based on the two parameters provided.
    ///
    pub fn new(nodes: &Vec<AM<Node>>,
               _training_ground_truths: &[f64],
               loss_function: Box<Fn(Vec<f64>, Vec<f64>) -> f64>) -> OutputLayer {
        let mut output_nodes = vec![];
        for node in nodes {
            output_nodes.push(node.clone())
        }

        let node_count = output_nodes.len();

        assert!(_training_ground_truths.len() % node_count == 0, "training_vals.len() must be a multiple of nodes.len()!");

        let mut training_ground_truths =
            Array2::<f64>::zeros((_training_ground_truths.len() / node_count, node_count));

        for (idx, val) in _training_ground_truths.iter().enumerate() {
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
    /// Make sure `InputLayer.set_iteration(iter)` is called with the same `iter` value first!
    ///
    /// The `iter` parameter is a ring which wraps around 0 and `self.training_inputs.len()`
    /// E.g. assuming iter is 5, and `training_inputs` has 3 vectors of node input values,
    /// the nodes will be assigned to the values given by index 2 (5 % 3) of `training_inputs`.
    pub fn calculate_iter_loss(&self, iter: usize) -> f64 {

        let idx = iter % self.training_ground_truths.len();

        let vals = self.training_ground_truths.slice(s![idx, ..]);
        let mut output_node_activations = vec![];

        for (idx, node_ground_truth_val) in vals.iter().enumerate() {
            let mut node = self.output_nodes[idx].lock().unwrap();

            let activation = node.calc_activation();
            output_node_activations.push(activation);
            println!("activation: {}, ground: {}", activation, node_ground_truth_val);
        }

        let mut expected_ground_truths: Vec<f64> = vals.to_vec();

        assert_eq!(output_node_activations.len(), expected_ground_truths.len(), "Unexpected error!!??");

        (self.loss_function)(output_node_activations, expected_ground_truths)

    }
}
