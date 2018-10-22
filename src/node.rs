//!
//! Just some basic nodes
//!

use std::f64;

/// The generic node trait
pub trait Node {
    /// 'Activation' refers to the output value of the node
    fn calc_activation(&self) -> f64;
}

pub struct InputNode {
    pub value: f64
}

impl InputNode {
    pub fn new(value: f64) -> InputNode {
        InputNode {
            value
        }
    }
}

impl Node for InputNode {
    fn calc_activation(&self) -> f64 {
        self.value
    }
}

/// Pass this node with const_value = 1 as input to whichever non-input nodes
/// that requires a bias. Essentially just an InputNode that
/// has a constant value
pub struct ConstantNode {
    pub const_value: f64
}

impl ConstantNode {
    pub fn new(const_value: f64) -> ConstantNode {
        ConstantNode {
            const_value
        }
    }
}

impl Node for ConstantNode {
    fn calc_activation(&self) -> f64 {
        self.const_value
    }
}

pub struct NodeWeight<'n> {
    pub node: &'n Node,
    pub weight: f64
}

/// Sums up all the products of each input-weight pair and passes
/// the result through a sigmoid logistic function.
pub struct SigmoidNode<'i> {
    pub inputs: Vec<NodeWeight<'i>>
}

impl <'i> Node for SigmoidNode<'i> {
    fn calc_activation(&self) -> f64 {
        let sum = self.inputs.iter().fold(0.0, |acc, x| {
            acc + x.node.calc_activation() * x.weight
        });

        let sigmoid_activation= 1.0 / (1.0 + f64::exp(-sum));

        sigmoid_activation
    }
}