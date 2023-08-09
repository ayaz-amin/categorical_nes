mod rng;
mod dist;
use crate::dist::Distribution;

struct Solution
{
    x1: i32,
    x2: i32,
    x3: i32,
    x4: i32,
    x5: i32
}

fn get_grad(dist: &dist::Categorical, sample: i32, score: f32) -> Vec<f32>
{
    let mut scored_grad: Vec<f32> = Vec::new();
    
    for i in 0..dist.logits.len()
    {
        let prob = dist.log_prob(sample).exp();
        if i == (sample as usize)
        {
            scored_grad.push(prob * (1.0 - prob) * score);
        }

        else
        {
            scored_grad.push(-prob * dist.log_prob(i as i32) * score);
        }
    }

    return scored_grad;
}

fn add_grad(grad1: &mut Vec<f32>, grad2: &Vec<f32>)
{
    for i in 0..grad1.len()
    {
        grad1[i] += grad2[i]
    }
}

fn update(dist: &mut dist::Categorical, grad: &Vec<f32>, num_muts: i32)
{
    for i in 0..dist.logits.len()
    {
        dist.logits[i] -= 0.01 * (grad[i] / num_muts as f32);
    }
}

fn get_max(dist: &dist::Categorical) -> i32
{
    let mut i = 0;
    let mut score = dist.log_prob(0);
    
    for j in 1..dist.logits.len()
    {
        if dist.log_prob(j as i32) > score
        {
            score = dist.log_prob(j as i32);
            i = j;
        }
    }

    i as i32
}

fn main() {
    let mut rng = rng::RNG::new(0);
    
    let mut dist1 = dist::Categorical::new(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut dist2 = dist::Categorical::new(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut dist3 = dist::Categorical::new(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut dist4 = dist::Categorical::new(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut dist5 = dist::Categorical::new(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    
    let sol = Solution{x1: 3, x2: 1, x3: 4, x4: 4, x5: 2};

    let num_iters = 1000;
    let num_muts = 50;

    for _i in 0..num_iters
    {
        let mut grad_x1 = Vec::new();
        let mut grad_x2 = Vec::new();
        let mut grad_x3 = Vec::new();
        let mut grad_x4 = Vec::new();
        let mut grad_x5 = Vec::new();
        
        for _j in 0..5
        {
            grad_x1.push(0.0);
            grad_x2.push(0.0);
            grad_x3.push(0.0);
            grad_x4.push(0.0);
            grad_x5.push(0.0);
        }

        let mut total_score = 0.0;
        for _k in 0..num_muts
        {
            let x1 = dist1.sample(&mut rng);
            let x2 = dist2.sample(&mut rng);
            let x3 = dist3.sample(&mut rng);
            let x4 = dist4.sample(&mut rng);
            let x5 = dist5.sample(&mut rng);

            let mse_x1 = (sol.x1 - x1).pow(2);
            let mse_x2 = (sol.x2 - x2).pow(2);
            let mse_x3 = (sol.x3 - x3).pow(2);
            let mse_x4 = (sol.x4 - x4).pow(2);
            let mse_x5 = (sol.x5 - x5).pow(2);

            total_score = ((mse_x1 + mse_x2 + mse_x3 + mse_x4 + mse_x5) as f32);
            let scored_grad_x1 = get_grad(&dist1, x1, total_score);
            let scored_grad_x2 = get_grad(&dist2, x2, total_score);
            let scored_grad_x3 = get_grad(&dist3, x3, total_score);
            let scored_grad_x4 = get_grad(&dist4, x4, total_score);
            let scored_grad_x5 = get_grad(&dist5, x5, total_score);

            add_grad(&mut grad_x1, &scored_grad_x1);
            add_grad(&mut grad_x2, &scored_grad_x2);
            add_grad(&mut grad_x3, &scored_grad_x3);
            add_grad(&mut grad_x4, &scored_grad_x4);
            add_grad(&mut grad_x5, &scored_grad_x5);
        }

        update(&mut dist1, &grad_x1, num_muts);
        update(&mut dist2, &grad_x2, num_muts);
        update(&mut dist3, &grad_x3, num_muts);
        update(&mut dist4, &grad_x4, num_muts);
        update(&mut dist5, &grad_x5, num_muts);

        println!("Score: {}", total_score);
    }

    let x1 = get_max(&dist1);
    let x2 = get_max(&dist2);
    let x3 = get_max(&dist3);
    let x4 = get_max(&dist4);
    let x5 = get_max(&dist5);
    println!("Expected: {}, {}, {}, {}, {}", sol.x1, sol.x2, sol.x3, sol.x4, sol.x5);
    println!("Inferred: {}, {}, {}, {}, {}", x1, x2, x3, x4, x5);
}
