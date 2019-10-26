mod layers;
mod network;
mod node;

#[macro_use]
extern crate lazy_static;
extern crate rand;
#[macro_use(s)]
extern crate ndarray;

use rand::Rng;
use std::sync::{Arc, Mutex};

use layers::{InputLayer, OutputLayer};
use network::Network;
use node::*;

/// Arc Mutex helpers because garbage collection
pub type AM<T> = Arc<Mutex<T>>;

pub fn am<T>(x: T) -> Arc<Mutex<T>> {
    Arc::new(Mutex::new(x))
}

fn main() {
    println!("rusty-brain v0.1: a + 2b test");

    let mut training_vals = vec![];
    let mut ground_truths = vec![];

    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let a = rng.gen_range(0f64, 5f64);
        let b = rng.gen_range(0f64, 5f64);
        training_vals.push(a);
        training_vals.push(b);

        ground_truths.push(a + 2.0 * b);
    }

    // Make inputs
    let mut i1 = InputNode::new("i1", 0.6);
    let mut i2 = InputNode::new("i2", 1.0);

    let mut input_layer = InputLayer::new(&vec![i1.clone(), i2.clone()], &training_vals);

    let mut s1 = SumNode::new("s1");

    connect_init(i1.clone(), s1.clone(), 1.0);
    connect_init(i2.clone(), s1.clone(), 0.2);

    let mut output_layer = OutputLayer::new(
        &vec![s1],
        &ground_truths,
        // Loss Function: Root Mean Square Error
        Box::new(|node_activations: Vec<f64>, ground_truths: Vec<f64>| {
            let mut mse = 0.0;

            let actual_expected = node_activations.iter().zip(ground_truths.iter());

            for (actual, expected) in actual_expected {
                mse += f64::powi(actual - expected, 2) / (node_activations.len() as f64);
            }

            mse
        }),
    );

    let mut network = Network::new(input_layer, output_layer);
    let output_node_count = network.output_layer.output_nodes.len() as f64;
    for iter in 0..=10 {
        network.input_layer.set_iteration(iter);
        let loss = network.output_layer.calculate_iter_loss(iter);
        println!("Iteration {}: loss = {}", iter, loss);
        let gt = network.output_layer.get_ground_truths(iter);

        network.evaluate_gradients(iter as i32, move |node_name| {
            // Lambda to calculate one derivative term of loss of one particular node.
            // This derivative function is assuming the RMSE function is used.
            // RMSE = sigma((node, k=10) ==> RMSE(node))
            // RMSE(node) = (node.activation - node.ground_truth) ^ 2
            // RMSE'(node) = 2 * (node.activation - node.ground_truth)
            let nodes = NODES.lock().unwrap();
            let node = nodes.get(node_name).unwrap();
            let node = node.lock().unwrap();
            2.0 * (node.get_last_calc_activation() - gt.get(node_name).unwrap()) / output_node_count
        })
    }
}
