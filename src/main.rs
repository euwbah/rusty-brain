mod network;
mod node;
mod layers;

#[macro_use]
extern crate lazy_static;
extern crate rand;
#[macro_use(s)]
extern crate ndarray;

use std::sync::{Arc, Mutex};
use rand::Rng;

use node::*;
use layers::{InputLayer, OutputLayer};
use network::Network;

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

    let mut input_layer =
        InputLayer::new(
            &vec![i1.clone(), i2.clone()],
            &training_vals,
        );

    let mut s1 = SumNode::new("s1");

    connect_init(i1.clone(), s1.clone(), 1.0);
    connect_init(i2.clone(), s1.clone(), 0.2);
    {
        let mut s1 = s1.lock().unwrap();

        for i in 0..10 {
            input_layer.set_iteration(i);
            println!("s1 activation: {}", s1.calc_activation());
        }
    }

    input_layer.set_iteration(0);

    let mut output_layer =
        OutputLayer::new(
            &vec![s1], &training_vals,

            // Loss Function: Root Mean Square Error
            Box::new(|node_activations: Vec<f64>, ground_truths: Vec<f64>| {
                let mut rmse = 0.0;

                let actual_expected = node_activations.iter().zip(ground_truths.iter());

                for (actual, expected) in actual_expected {
                    rmse += f64::powi(actual - expected, 2);
                }

                rmse
            }));

    let iter0_loss = output_layer.calculate_iter_loss(0);

    println!("iter0_loss: {}", iter0_loss);

    let network = Network::new(input_layer, output_layer);
}
