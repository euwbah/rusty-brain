//!
//! Just some basic nodes
//!

use std::f64;
use rand::prelude::*;
use AM;
use am;

/// The generic node trait
pub trait Node {
    /// 'Activation' refers to the output value of the node
    fn calc_activation(&self) -> f64;

    /// Retrieve a node's unique identifier
    fn name(&self) -> &str;

    /// Updates weights of input nodes (if any) based on `gradient` and input node value.
    ///
    /// Then, recurse for each child input node with the `gradient` parameter set to
    /// d(loss) / d(child input node activation output).
    ///
    /// `gradient` represents the value of d(loss) / d(node activation output).
    /// (i.e., how much the loss will change if this node's output were to change by some small value d)
    fn train(&mut self, gradient: f64);

    /// Get a list of nodes connected as inputs of this node.
    fn input_nodes(&self) -> Vec<AM<Node>>;

    /// Get a list of nodes receiving this node's output as a parameter.
    /// Used for calculating the derivative of all the nodes recursively.
    /// No mutability needed, hopefully!
    fn output_nodes(&self) -> &Vec<AM<Node>>;

    /// Register a node as an input for this node
    /// Do not call this function on its own, use the `connect` function instead
    fn add_input_node(&mut self, input_node: AM<Node>);

    fn add_input_node_init(&mut self, input_node: AM<Node>, weight: f64);

    /// Register a node as a receiver of the output from this node
    /// Do not call this function on its own, use the `connect` function instead
    fn add_output_node(&mut self, output_node: AM<Node>);
}

/// Connect the output of node a to the input of node b.
///
/// Note the following method is required as it is not possible to assign references
/// from A to B and from B to A simultaneously when in the scope of either A or B.
/// Hence, a function outside the scope of A or B's `self` is required as only then
/// can A and B reference each other.
pub fn connect(a: AM<Node>, b: AM<Node>) {
    a.lock().unwrap().add_output_node(b.clone());
    b.lock().unwrap().add_input_node(a);
}

/// Connect the output of node a to the input of node b with a preset weight.
pub fn connect_init(a: AM<Node>, b: AM<Node>, weight: f64) {
    a.lock().unwrap().add_output_node(b.clone());
    b.lock().unwrap().add_input_node_init(a, weight);
}

pub struct InputNode {
    pub name: &'static str,
    pub value: f64,
    outputs: Vec<AM<Node>>,
}

impl InputNode {
    pub fn new(name: &'static str, value: f64) -> InputNode {
        InputNode {
            name,
            value,
            outputs: vec![],
        }
    }
}

impl Node for InputNode {
    fn calc_activation(&self) -> f64 {
        self.value
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn train(&mut self, gradient: f64) {
        // Nothing to train
    }

    fn input_nodes(&self) -> Vec<AM<Node>> {
        vec![]
    }

    fn output_nodes(&self) -> &Vec<AM<Node>> {
        &self.outputs
    }

    fn add_input_node(&mut self, input_node: AM<Node>) {
        panic!("InputNode does not have an input!")
    }

    fn add_input_node_init(&mut self, input_node: AM<Node>, weight: f64) {
        panic!("InputNode does not have an input!")
    }

    fn add_output_node(&mut self, node: AM<Node>) {
        self.outputs.push(node);
    }
}

/// Pass this node with const_value = 1 as input to whichever non-input nodes
/// that requires a bias. Essentially just an InputNode that
/// has a constant value
pub struct ConstantNode {
    pub name: &'static str,
    pub const_value: f64,
    outputs: Vec<AM<Node>>,
}

impl ConstantNode {
    pub fn new(name: &'static str, const_value: f64) -> ConstantNode {
        ConstantNode {
            name,
            const_value,
            outputs: vec![],
        }
    }
}

impl Node for ConstantNode {
    fn calc_activation(&self) -> f64 {
        self.const_value
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn train(&mut self, gradient: f64) {
        // Nothing to train
    }

    fn input_nodes(&self) -> Vec<AM<Node>> {
        vec![]
    }

    fn output_nodes(&self) -> &Vec<AM<Node>> {
        &self.outputs
    }

    fn add_input_node(&mut self, input_node: AM<Node>) {
        panic!("ConstantNode does not have an input!")
    }

    fn add_input_node_init(&mut self, input_node: AM<Node>, weight: f64) {
        panic!("ConstantNode does not have an input!")
    }

    fn add_output_node(&mut self, node: AM<Node>) {
        self.outputs.push(node);
    }
}

pub struct NodeWeight {
    pub node: AM<Node>,
    pub weight: f64,
}

impl NodeWeight {
    pub fn new(node: AM<Node>, weight: f64) -> NodeWeight {
        NodeWeight {
            node,
            weight,
        }
    }

    pub fn calc_weighted_activation(&self) -> f64 {
        self.node.lock().unwrap().calc_activation() * self.weight
    }
}

/// Sums up all the products of each input-weight pair
pub struct SumNode {
    pub name: &'static str,
    pub inputs: Vec<NodeWeight>,
    outputs: Vec<AM<Node>>,
}

impl SumNode {
    pub fn new(name: &'static str) -> SumNode {
        SumNode {
            name,
            inputs: vec![],
            outputs: vec![],
        }
    }
}

impl Node for SumNode {
    fn calc_activation(&self) -> f64 {
        let sum = self.inputs.iter().fold(0.0, |acc, node_weight| {
            acc + node_weight.calc_weighted_activation()
        });

        sum
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Updates weights of input nodes (if any) based on `gradient` and input node value.
    ///
    /// `gradient` represents the value of d(loss) / d(node activation output).
    /// (i.e., how much the loss will change if this node's output were to change by some small value d)
    fn train(&mut self, gradient: f64) {}

    fn input_nodes<'a>(&'a self) -> Vec<AM<Node>> {
        self.inputs.iter().map(|x| x.node.clone()).collect()
    }

    fn output_nodes(&self) -> &Vec<AM<Node>> {
        &self.outputs
    }

    /// Add an input with a randomly initialized weight ranging from -1 to 1
    /// DO NOT CALL ALONE. Use `connect()` instead
    fn add_input_node(&mut self, input_node: AM<Node>) {
        self.add_input_node_init(input_node, thread_rng().gen_range(-1.0, 1.0));
    }

    fn add_input_node_init(&mut self, input_node: AM<Node>, weight: f64) {
        self.inputs.push(NodeWeight::new(input_node, weight));
    }

    fn add_output_node(&mut self, node: AM<Node>) {
        self.outputs.push(node);
    }
}

/// Sums up all the products of each input-weight pair and passes
/// the result through a sigmoid logistic function.
pub struct SigmoidNode {
    pub name: &'static str,
    pub inputs: Vec<NodeWeight>,
    outputs: Vec<AM<Node>>,
}

impl Node for SigmoidNode {
    fn calc_activation(&self) -> f64 {
        let sum = self.inputs.iter().fold(0.0, |acc, x| {
            acc + x.node.lock().unwrap().calc_activation() * x.weight
        });

        let sigmoid_activation = 1.0 / (1.0 + f64::exp(-sum));

        sigmoid_activation
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn train(&mut self, gradient: f64) {}

    fn input_nodes(&self) -> Vec<AM<Node>> {
        self.inputs.iter().map(|x| x.node.clone()).collect()
    }

    fn output_nodes(&self) -> &Vec<AM<Node>> {
        &self.outputs
    }

    /// Add an input with a randomly initialized weight ranging from -1 to 1
    /// DO NOT CALL ALONE. Use `connect()` instead
    fn add_input_node(&mut self, input_node: AM<Node>) {
        self.add_input_node_init(input_node, thread_rng().gen_range(-1.0, 1.0));
    }

    fn add_input_node_init(&mut self, input_node: AM<Node>, weight: f64) {
        self.inputs.push(NodeWeight::new(input_node, weight));
    }

    fn add_output_node(&mut self, node: AM<Node>) {
        self.outputs.push(node);
    }
}

