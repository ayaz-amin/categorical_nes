use std::f32::consts::TAU;
use crate::rng::RNG;

pub trait Distribution
{
    type SampleType;
    type GradType;
    fn sample(&self, rng: &mut RNG) -> Self::SampleType;
    fn argmax(&self) -> Self::SampleType;
    fn log_prob(&self, x: Self::SampleType) -> f32;
    fn grad(&self, traces: Vec<(Self::SampleType, f32)>) -> Self::GradType;
    fn update(&mut self, grad: Self::GradType, rate: f32);
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
    type GradType = f32;

    fn sample(&self, rng: &mut RNG) -> Self::SampleType
    {
        let u1: f32 = rng.sample();
        let u2: f32 = rng.sample();
        let mag: f32 = self.stddev * (-2.0 * u1.ln()).sqrt();
        mag * (TAU * u2).cos() + self.mean
    }

    fn argmax(&self) -> Self::SampleType
    {
        self.mean
    }

    fn log_prob(&self, x: Self::SampleType) -> f32
    {
        const HALF_LN_TAU: f32 = 0.91893853320467274178032973640562;
        let z = (x - self.mean) / self.stddev;
        -self.stddev.ln() - HALF_LN_TAU - 0.5 * z.powf(2.0)
    }

    fn grad(&self, traces: Vec<(Self::SampleType, f32)>) -> Self::GradType
    {
        let mut grad: f32 = 0.0;
        let num_mutations: f32 = traces.len() as f32;
        for i in 0..traces.len()
        {
            let (sample, score) = traces[i];
            let normalized = (sample - self.mean) / self.stddev;
            grad += normalized * score;
        }

        grad /= self.stddev * num_mutations;
        grad
    }

    fn update(&mut self, grad: Self::GradType, rate: f32)
    {
        self.mean -= rate * self.stddev * grad;
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
    type SampleType = usize;
    type GradType = Vec<f32>;

    fn sample(&self, rng: &mut RNG) -> Self::SampleType
    {
        let mut score: f32 = self.logits[0];
        let mut index: usize = 0;

        for i in 1..self.logits.len()
        {
            let u: f32 = rng.sample();
            let gumbel: f32 = -(-u.ln()).ln();
            let sample: f32 = self.logits[i] + gumbel;
            if sample > score
            {
                score = sample;
                index = i;
            }
        }

        index
    }

    fn argmax(&self) -> Self::SampleType
    {
        let mut score: f32 = self.logits[0];
        let mut index: usize = 0;

        for i in 1..self.logits.len()
        {
            let sample: f32 = self.logits[i];
            if sample > score
            {
                score = sample;
                index = i;
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

    fn grad(&self, traces: Vec<(Self::SampleType, f32)>) -> Self::GradType
    {
        let mut grad: Vec<f32> = vec![0.0; self.logits.len()];
        let num_mutations: f32 = traces.len() as f32;
        
        for i in 0..traces.len()
        {
            let (sample, score) = traces[i];
            let mut scored_grad: f32 = 0.0;
            let prob = self.log_prob(sample).exp();
        
            for j in 0..self.logits.len()
            {
                let prob_j = self.log_prob(j).exp();
                if j == sample
                {
                    scored_grad = prob * (1.0 - prob) * score;
                }

                else
                {
                    scored_grad = -prob * prob_j * score;
                }
                
                grad[j] += scored_grad;
            }
        }

        for i in 0..grad.len()
        {
            let prob_i = self.log_prob(i).exp();
            grad[i] *= prob_i;
            grad[i] /= num_mutations;
        }

        grad
    }

    fn update(&mut self, grad: Self::GradType, rate: f32)
    {
        for i in 0..self.logits.len()
        {
            self.logits[i] -= rate * grad[i];
        }
    }
}