use std::f32::consts::TAU;
use crate::rng::RNG;

pub trait Distribution
{
    type SampleType;
    fn sample(&self, rng: &mut RNG) -> Self::SampleType;
    fn log_prob(&self, x: Self::SampleType) -> f32;
}

// Normal distribution
pub struct Normal
{
    pub mean: f32,
    pub stddev: f32
}

impl Normal
{
    pub fn new(mean: f32, stddev: f32) -> Self
    {
        Self {mean, stddev}
    }
}

impl Distribution for Normal
{
    type SampleType = f32;

    fn sample(&self, rng: &mut RNG) -> Self::SampleType
    {
        let u1: f32 = rng.sample();
        let u2: f32 = rng.sample();
        let mag: f32 = self.stddev * (-2.0 * u1.ln()).sqrt();
        mag * (TAU * u2).cos() + self.mean
    }

    fn log_prob(&self, x: Self::SampleType) -> f32
    {
        const HALF_LN_TAU: f32 = 0.91893853320467274178032973640562;
        let z = (x - self.mean) / self.stddev;
        -self.stddev.ln() - HALF_LN_TAU - 0.5 * z.powf(2.0)
    }
}

// Categorical distribution
pub struct Categorical
{
    pub logits: Vec<f32>
}

impl Categorical
{
    pub fn new(logits: Vec<f32>) -> Self
    {
        Self {logits}
    }
}

impl Distribution for Categorical
{
    type SampleType = i32;

    fn sample(&self, rng: &mut RNG) -> Self::SampleType
    {
        let mut score: f32 = self.logits[0];
        let mut index: i32 = 0;

        for i in 1..self.logits.len()
        {
            let u: f32 = rng.sample();
            let gumbel: f32 = -(-u.ln()).ln();
            let sample: f32 = self.logits[i] + gumbel;
            if sample > score
            {
                score = sample;
                index = i as i32;
            }
        }

        index
    }

    fn log_prob(&self, x: Self::SampleType) -> f32
    {
        let mut max_val: f32 = self.logits[0];
        for i in 1..self.logits.len()
        {
            if self.logits[i] > max_val
            {
                max_val = self.logits[i];
            }
        }

        let mut sum: f32 = 0.0;
        for logit in self.logits.iter()
        {
            sum += (logit - max_val).exp();
        }
        
        (self.logits[x as usize] - max_val) - sum.ln()
    }
}