//!
//! Just some basic nodes
//!

use std::f64;
use std::sync::Mutex;
use std::collections::HashMap;
use rand::prelude::*;
use AM;
use am;

lazy_static! {
    /// Global lookup for nodes
    pub static ref NODES: Mutex<HashMap<String, AM<Node + Send>>> = Mutex::new(HashMap::new());
}

/// Call this in the constructor of nodes
fn register_node(name: &str, node: AM<Node + Send>) {
    if let Some(_) = NODES.lock().unwrap().insert(name.to_string(), node) {
        panic!("Cannot create two nodes with same name! [{}]", name);
    }
}

/// This object is passed as a constant parameter through the recursive
/// `calc_activation_derivative` function.
pub struct DerivativeCalculationParams {
    /// The calculation iteration
    calc_derivative_iteration: i32,
    /// Node name: partial derivative of loss function
    ///
    /// This method is used when the loss function can be expressed as a sum
    /// of single-argument functions, e.g., loss = f(a) + f(b) + f(c) + etc...
    /// where a, b, c, ... are activation values of final layer nodes, as in this case
    /// the derivative of the loss function against each output node's activation value
    /// can be simply expressed as f'(a), f'(b), f'(c), etc...
    ///
    /// The values in this hashmap represent the output of the functions f'(a), f'(b), f'(c), etc...
    output_nodes_loss_fn_derivative: HashMap<String, f64>,
}

impl DerivativeCalculationParams {
    pub fn new<F>(calc_derivative_iteration: i32,
               output_layer_node_names: Vec<String>,
               derivative_fn: F) -> DerivativeCalculationParams
    where F: 'static + Fn(&str) -> f64 {
        let mut output_nodes_loss_fn_derivative: HashMap<String, f64> = HashMap::new();

        for n in output_layer_node_names {
            let n_clone = n.clone();
            output_nodes_loss_fn_derivative.insert(
                n.clone(),
            derivative_fn(n_clone.as_str()));
        }

        DerivativeCalculationParams {
            calc_derivative_iteration,
            output_nodes_loss_fn_derivative
        }
    }
}

/// The generic node trait
pub trait Node {
    /// Retrieve a node's unique identifier
    fn name(&self) -> &str;

    /// 'Activation' refers to the output value of the node.
    ///
    /// This should also update the stored activation value which will be returned by
    /// `get_last_calc_activation()`
    fn calc_activation(&mut self) -> f64;

    /// Retrieves the last calculated activation.
    ///
    /// This cached value should be updated every time `calc_activation` is called. However,
    /// there is no need to call `calc_activation` on each of the nodes directly, instead
    /// calling `calculate_iter_loss()` on the `OutputLayer` object will make all connected nodes
    /// update their cached activation values.
    ///
    /// This is useful when trying to calculate node determinants and the node
    /// activation value is needed such as
    /// d(sigmoid(z)) / d(z).
    fn get_last_calc_activation(&self) -> f64;

    /// Returns (Iteration no. of last derivative calculation, last derivative calculation value)
    fn get_training_state(&self) -> &TrainingState;

    fn get_training_state_mut(&mut self) -> &mut TrainingState;

    /// Calculates the derivative of the loss function against
    /// this node's activation value (as per `get_last_calc_activation()`), then stores the
    /// derivative for later use when updating the nodes.
    ///
    /// Note that the cached activation values from `get_last_calc_activation()` is updated
    /// by calling
    ///
    /// This is a recursive function which starts off by invoking this method on all
    /// input nodes, then follows by calling this method on all of its `output_nodes()`.
    /// Of course, this will lead to numerous unnecessary recalculations of nodes that share the
    /// same set of multiple children (such as in fully-connected networks)
    ///
    /// As such, a derivative calculation iteration counter is stored in the DerivativeCalculationParams
    /// parameter which should be compared to a local DerivativeCalculationParams store which updates per call,
    /// if this function has been called on the same object twice, the calculation iteration id
    /// would have been found to be the same and the derivative calculation and recursion of
    /// its output nodes can be skipped.
    fn calc_activation_derivative(&mut self, calc_state: &DerivativeCalculationParams) -> f64 {
        if self.get_training_state().calc_derivative_iteration != calc_state.calc_derivative_iteration {
            /*
                Simply sum up partial derivatives of each output node.
                Let x            -> this activation
                Let a, b, c, ... -> three output nodes receiving this node activation as input

                d(loss) / d(self.activation) = da/dx * dL/da + db/dx * dL/db + dc/dx * dL/dc + etc...
            */

            self.get_training_state_mut().dloss = 0.0;

            let output_nodes_count = {
                self.output_nodes().len()
            };

            if output_nodes_count != 0 {
                let mut final_dloss = 0.0;
                for o in self.output_nodes() {
                    let mut o = o.lock().unwrap();
                    let dloss_partial_derivative =
                        o.calc_derivative_against(self.name()) * o.calc_activation_derivative(&calc_state);

                    final_dloss += dloss_partial_derivative;
                }

                self.get_training_state_mut().dloss = final_dloss;
            } else if let Some(derivative) = calc_state.output_nodes_loss_fn_derivative.get(self.name()) {
                // If there are no output nodes, check calc_state if this node is an output node
                // with a given partial loss function

                self.get_training_state_mut().dloss = *derivative;
            } else {
                println!("WARNING: [{}] Last layer node found that doesn't have a registered loss \
                function partial derivative, defaulting gradient to 0.", self.name());
            }
        }

        println!("dLoss/d[{}]: {}", self.name(), self.get_training_state().dloss);

        self.get_training_state_mut().dloss
    }

    /// Calculates the value of d(self activation) / d(input_node activation)
    /// assuming that `input_node` is an immediate input to the input_node
    ///
    /// This is the consumer function for the recursive `calc_activation_derivative` which
    /// steps the recursion forward.
    fn calc_derivative_against(&self, input_node_name: &str) -> f64;

    /// Updates weights of input nodes (if any) based on `gradient` and input node value.
    ///
    /// Then, recurse for each child input node with the `gradient` parameter set to
    /// d(loss) / d(child input node activation output).
    ///
    /// `gradient` represents the value of d(loss) / d(node activation output).
    /// (i.e., how much the loss will change if this node's output were to change by some small value d)
    fn update_weights(&mut self, step_size: f64) {
        // default to no weights to update
    }

    /// Get a list of nodes connected as inputs of this node.
    fn input_nodes(&self) -> Vec<AM<Node + Send>>;

    fn input_node_weights(&self) -> AM<HashMap<String, NodeWeight>>;

    /// Get a list of nodes receiving this node's output as a parameter.
    /// Used for calculating the derivative of all the nodes recursively.
    /// No mutability needed, hopefully!
    fn output_nodes(&self) -> &Vec<AM<Node + Send>>;

    /// Register a node as an input for this node
    /// Do not call this function on its own, use the `connect` function instead
    fn add_input_node(&mut self, input_node: AM<Node + Send>);

    /// Register a node as an input for this node with a predefined weight
    /// Do not call this function on its own, use the `connect` function instead
    fn add_input_node_init(&mut self, input_node: AM<Node + Send>, weight: f64);

    /// Register a node as a receiver of the output from this node
    /// Do not call this function on its own, use the `connect` function instead
    fn add_output_node(&mut self, output_node: AM<Node + Send>);
}

/// Connect the output of node a to the input of node b.
///
/// Note the following method is required as it is not possible to assign references
/// from A to B and from B to A simultaneously when in the scope of either A or B.
/// Hence, a function outside the scope of A or B's `self` is required as only then
/// can A and B reference each other.
pub fn connect(a: AM<Node + Send>, b: AM<Node + Send>) {
    a.lock().unwrap().add_output_node(b.clone());
    b.lock().unwrap().add_input_node(a);
}

/// Connect the output of node a to the input of node b with a preset weight.
pub fn connect_init(a: AM<Node + Send>, b: AM<Node + Send>, weight: f64) {
    a.lock().unwrap().add_output_node(b.clone());
    b.lock().unwrap().add_input_node_init(a, weight);
}

/// Contains stateful data used by all nodes during training
struct TrainingState {
    /// The value of `DerivativeCalculationParams.calc_derivative_iteration` when
    /// `Node.calc_activation_derivative()` was last called.
    calc_derivative_iteration: i32,
    /// The last value of d(loss)/d(this activation) as calculated by
    /// `Node.calc_activation_derivative()`.
    dloss: f64,
}

impl Default for TrainingState {
    fn default() -> Self {
        TrainingState {
            calc_derivative_iteration: -1,
            dloss: 0.0,
        }
    }
}

pub struct InputNode {
    pub name: String,
    pub value: f64,
    outputs: Vec<AM<Node + Send>>,
    training_state: TrainingState,
    /// Fighting borrow checker
    empty_hashmap: AM<HashMap<String, NodeWeight>>,
}

impl InputNode {
    pub fn new(_name: &str, value: f64) -> AM<InputNode> {
        let name = _name.to_string();
        let node = InputNode {
            name,
            value,
            outputs: vec![],
            training_state: Default::default(),
            empty_hashmap: am(HashMap::new()),
        };

        let node = am(node);

        register_node(_name, node.clone());

        node
    }
}

impl Node for InputNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_activation(&mut self) -> f64 {
        self.value
    }

    fn get_last_calc_activation(&self) -> f64 {
        self.value
    }

    fn get_training_state(&self) -> &TrainingState {
        &self.training_state
    }

    fn get_training_state_mut(&mut self) -> &mut TrainingState {
        &mut self.training_state
    }

    fn calc_derivative_against(&self, input_node_name: &str) -> f64 {
        panic!("Attempted to calculate derivative against an InputNode");
    }

    fn input_nodes(&self) -> Vec<AM<Node + Send>> {
        vec![]
    }

    fn input_node_weights(&self) -> AM<HashMap<String, NodeWeight>> {
        self.empty_hashmap.clone()
    }

    fn output_nodes(&self) -> &Vec<AM<Node + Send>> {
        &self.outputs
    }

    fn add_input_node(&mut self, input_node: AM<Node + Send>) {
        panic!("InputNode does not have an input!")
    }

    fn add_input_node_init(&mut self, input_node: AM<Node + Send>, weight: f64) {
        panic!("InputNode does not have an input!")
    }

    fn add_output_node(&mut self, node: AM<Node + Send>) {
        self.outputs.push(node);
    }
}

/// Pass this node with const_value = 1 as input to whichever non-input nodes
/// that requires a bias. Essentially just an InputNode that
/// has a constant value
pub struct ConstantNode {
    pub name: String,
    pub const_value: f64,
    outputs: Vec<AM<Node + Send>>,
    training_state: TrainingState,
    /// Fighting borrow checker
    empty_hashmap: AM<HashMap<String, NodeWeight>>,
}

impl ConstantNode {
    pub fn new(name: &str, const_value: f64) -> AM<ConstantNode> {
        let node = ConstantNode {
            name: name.to_string(),
            const_value,
            outputs: vec![],
            training_state: Default::default(),
            empty_hashmap: am(HashMap::new()),
        };
        let node = am(node);

        register_node(name, node.clone());

        node
    }
}

impl Node for ConstantNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_activation(&mut self) -> f64 {
        self.const_value
    }

    fn get_last_calc_activation(&self) -> f64 {
        self.const_value
    }

    fn get_training_state(&self) -> &TrainingState {
        &self.training_state
    }

    fn get_training_state_mut(&mut self) -> &mut TrainingState {
        &mut self.training_state
    }

    fn calc_derivative_against(&self, input_node_name: &str) -> f64 {
        panic!("Attempted to calculate derivative against a ConstantNode!");
    }

    fn input_nodes(&self) -> Vec<AM<Node + Send>> {
        vec![]
    }

    fn input_node_weights(&self) -> AM<HashMap<String, NodeWeight>> {
        self.empty_hashmap.clone()
    }

    fn output_nodes(&self) -> &Vec<AM<Node + Send>> {
        &self.outputs
    }

    fn add_input_node(&mut self, input_node: AM<Node + Send>) {
        panic!("ConstantNode does not have an input!")
    }

    fn add_input_node_init(&mut self, input_node: AM<Node + Send>, weight: f64) {
        panic!("ConstantNode does not have an input!")
    }

    fn add_output_node(&mut self, node: AM<Node + Send>) {
        self.outputs.push(node);
    }
}

pub struct NodeWeight {
    pub node: AM<Node + Send>,
    pub weight: f64,
}

impl NodeWeight {
    pub fn new(node: AM<Node + Send>, weight: f64) -> NodeWeight {
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
    pub name: String,
    /// Node name: NodeWeight
    pub inputs: AM<HashMap<String, NodeWeight>>,
    outputs: Vec<AM<Node + Send>>,
    /// Stores the last value returned by `calc_activation()`.
    /// Only updated when `calc_activation()` is called.
    activation: f64,
    training_state: TrainingState,
}

impl SumNode {
    pub fn new(name: &str) -> AM<SumNode> {
        let node = SumNode {
            name: name.to_string(),
            inputs: am(HashMap::new()),
            outputs: vec![],
            activation: 0.0,
            training_state: Default::default(),
        };

        let node = am(node);

        register_node(name, node.clone());

        node
    }
}

impl Node for SumNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_activation(&mut self) -> f64 {
        let sum = self.inputs.lock().unwrap().iter().fold(
            0.0,
            |acc, (name, node_weight)| {
                acc + node_weight.calc_weighted_activation()
            });

        self.activation = sum;

        sum
    }

    fn get_last_calc_activation(&self) -> f64 {
        self.activation
    }

    fn get_training_state(&self) -> &TrainingState {
        &self.training_state
    }

    fn get_training_state_mut(&mut self) -> &mut TrainingState {
        &mut self.training_state
    }

    fn calc_derivative_against(&self, input_node_name: &str) -> f64 {
        // since there is no activation function, derivative is just
        // d(weight * input_node activation) / d(input_node activation), i.e. just weight.

        self.inputs.lock().unwrap().get(input_node_name)
            .expect(format!("[{}] is not an input of [{}]", input_node_name, self.name).as_str())
            .weight
    }

    /// Updates weights of input nodes (if any) based on the previously calculated dloss.
    /// `step_size` represents the multiplier of the dloss derivative to adjust the weight by.
    fn update_weights(&mut self, step_size: f64) {
        /*
            let loss     --> loss score
                actv     --> activation of this node
                actv_bar --> activation of this node before passing through the activation function
                             (in the SumNode, the activation function is the identity function)
                             this is also known as the "weighted sum"
                weight   --> weight multiplier of an input node

            d(loss)/d(weight) = d(loss)/d(actv) * d(actv)/d(actv_bar) * d(actv_bar)/d(weight)

            d(loss)/d(actv) is already given as `dloss_dactv`
            d(actv)/d(actv_bar) is 1 as there is no activation function for the simple sum node.
                                the derivative of f(x) = x is 1.
            d(actv_bar)/d(weight) is the activation value of the input node,
                                  since actv_bar = input * weight,
                                  d(actv_bar)/d(weight) = input

        */

        let dloss_dactv = self.training_state.dloss;
        let dactv_dactv_bar = 1.0; // f(x) = x ==> f'(x) = 1, identity activation function

        let mut inputs_dloss = vec![];

        let mut inputs = self.inputs.lock().unwrap();
        for k in inputs.keys() {
            let mut nw = inputs.get(k).unwrap();
            let dactv_bar_weight = nw.node.lock().unwrap().get_last_calc_activation();

            let dloss_dweight = dloss_dactv * dactv_dactv_bar * dactv_bar_weight;

            inputs_dloss.push((k.to_owned(), dloss_dweight));
        }

        for (i, dloss) in inputs_dloss {
            inputs.get_mut(i.as_str()).unwrap().weight -= step_size * dloss;
        }
    }

    fn input_nodes<'a>(&'a self) -> Vec<AM<Node + Send>> {
        self.inputs.lock().unwrap().iter().map(|(_, x)| x.node.clone()).collect()
    }

    fn input_node_weights(&self) -> AM<HashMap<String, NodeWeight>> {
        self.inputs.clone()
    }

    fn output_nodes(&self) -> &Vec<AM<Node + Send>> {
        &self.outputs
    }

    /// Add an input with a randomly initialized weight ranging from -1 to 1
    /// DO NOT CALL ALONE. Use `connect()` instead
    fn add_input_node(&mut self, input_node: AM<Node + Send>) {
        self.add_input_node_init(input_node, thread_rng().gen_range(-1.0, 1.0));
    }

    fn add_input_node_init(&mut self, input_node: AM<Node + Send>, weight: f64) {
        let clone = input_node.clone();
        self.inputs.lock().unwrap().insert(clone.lock().unwrap().name().to_string(),
                                           NodeWeight::new(input_node, weight));
    }

    fn add_output_node(&mut self, node: AM<Node + Send>) {
        self.outputs.push(node);
    }
}

/// Sums up all the products of each input-weight pair and passes
/// the result through a sigmoid logistic function.
pub struct SigmoidNode {
    pub name: String,
    /// Node name: NodeWeight
    pub inputs: AM<HashMap<String, NodeWeight>>,
    outputs: Vec<AM<Node + Send>>,
    /// Stores the last value returned by `calc_activation()`.
    /// Only updated when `calc_activation()` is called.
    activation: f64,
    training_state: TrainingState,
}

impl Node for SigmoidNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn calc_activation(&mut self) -> f64 {
        let sum = self.inputs.lock().unwrap().iter().fold(
            0.0,
            |acc, (_, x)| {
                acc + x.node.lock().unwrap().calc_activation() * x.weight
            });

        let sigmoid_activation = 1.0 / (1.0 + f64::exp(-sum));

        self.activation = sigmoid_activation;

        sigmoid_activation
    }

    fn get_last_calc_activation(&self) -> f64 {
        self.activation
    }

    fn get_training_state(&self) -> &TrainingState {
        &self.training_state
    }

    fn get_training_state_mut(&mut self) -> &mut TrainingState {
        &mut self.training_state
    }

    fn calc_derivative_against(&self, input_node_name: &str) -> f64 {
        // let z -> input_node activation * connection weight
        // hence, dz/d(input activation) = w
        // let a -> sigmoid(z)
        // hence, da/dz = a(1 - a)
        // d(a) / d(input_node activation) = d(a)/d(z) * d(z)/d(input_node activation)
        //                                 = sigmoid(z)(1 - sigmoid(z)) * connection weight

        let w =
            self.inputs.lock().unwrap().get(input_node_name)
                .expect(format!("[{}] is not an input of [{}]", input_node_name, self.name).as_str())
                .weight;

        let a = self.get_last_calc_activation();

        a * (1.0 - a) * w
    }

    fn update_weights(&mut self, step_size: f64) {
        unimplemented!();
    }

    fn input_nodes(&self) -> Vec<AM<Node + Send>> {
        self.inputs.lock().unwrap().iter().map(|(_, x)| x.node.clone()).collect()
    }

    fn input_node_weights(&self) -> AM<HashMap<String, NodeWeight>> {
        self.inputs.clone()
    }

    fn output_nodes(&self) -> &Vec<AM<Node + Send>> {
        &self.outputs
    }

    /// Add an input with a randomly initialized weight ranging from -1 to 1
    /// DO NOT CALL ALONE. Use `connect()` instead
    fn add_input_node(&mut self, input_node: AM<Node + Send>) {
        self.add_input_node_init(input_node, thread_rng().gen_range(-1.0, 1.0));
    }

    fn add_input_node_init(&mut self, input_node: AM<Node + Send>, weight: f64) {
        let clone = input_node.clone();
        self.inputs.lock().unwrap().insert(clone.lock().unwrap().name().to_string(),
                                           NodeWeight::new(input_node, weight));
    }

    fn add_output_node(&mut self, node: AM<Node + Send>) {
        self.outputs.push(node);
    }
}

