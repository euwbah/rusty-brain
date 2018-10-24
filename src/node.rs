//!
//! Just some basic nodes
//!

use std::f64;
use rand::prelude::*;
use AM;

/// The generic node trait
pub trait Node {
    /// 'Activation' refers to the output value of the node
    fn calc_activation(&self) -> f64;

    fn name(&self) -> &str;
}

pub struct InputNode {
    pub name: &'static str,
    pub value: f64
}

impl InputNode {
    pub fn new(name: &'static str, value: f64) -> InputNode {
        InputNode {
            name,
            value
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
}

/// Pass this node with const_value = 1 as input to whichever non-input nodes
/// that requires a bias. Essentially just an InputNode that
/// has a constant value
pub struct ConstantNode {
    pub name: &'static str,
    pub const_value: f64,
}

impl ConstantNode {
    pub fn new(name: &'static str, const_value: f64) -> ConstantNode {
        ConstantNode {
            name,
            const_value
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
}

pub struct NodeWeight {
    pub node: AM<Node>,
    pub weight: f64
}

impl NodeWeight {
    pub fn new(node: AM<Node>, weight: f64) -> NodeWeight {
        NodeWeight {
            node, weight
        }
    }
}

/// Sums up all the products of each input-weight pair
pub struct SumNode {
    pub name: &'static str,
    pub inputs: Vec<NodeWeight>
}

impl SumNode {
    pub fn new(name: &'static str) -> SumNode {
        SumNode {
            name,
            inputs: vec![]
        }
    }

    /// Add an input with a randomly initialized weight ranging from 0 to 1
    pub fn add_input(&mut self, node: AM<Node>) {
        self.inputs.push( NodeWeight::new(node, thread_rng().gen_range(0.0, 1.0)));
    }

    /// Add an input with a preset weight
    pub fn add_input_init(&mut self, node: AM<Node>, weight: f64) {
        self.inputs.push( NodeWeight::new(node, weight));
    }

}

impl Node for SumNode {
    fn calc_activation(&self) -> f64 {
        let sum = self.inputs.iter().fold(0.0, |acc, x| {
            acc + x.node.lock().unwrap().calc_activation() * x.weight
        });

        sum
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Sums up all the products of each input-weight pair and passes
/// the result through a sigmoid logistic function.
pub struct SigmoidNode {
    pub name: &'static str,
    pub inputs: Vec<NodeWeight>
}

impl Node for SigmoidNode {
    fn calc_activation(&self) -> f64 {
        let sum = self.inputs.iter().fold(0.0, |acc, x| {
            acc + x.node.lock().unwrap().calc_activation() * x.weight
        });

        let sigmoid_activation= 1.0 / (1.0 + f64::exp(-sum));

        sigmoid_activation
    }

    fn name(&self) -> &str {
        &self.name
    }
}

