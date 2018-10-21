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

/// Pass this node as input to whichever non-input nodes
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


/// Sums up all the products of each input-weight pair and passes
/// the result through a sigmoid logistic function.
pub struct SigmoidNode<'i> {
    pub inputs: Vec<&'i mut Node>
}

impl <'i> Node for SigmoidNode<'i> {
    fn calc_activation(&self) -> f64 {
        0.0
    }
}