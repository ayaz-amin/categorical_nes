use crate::rng;
use plotters::prelude::*;

use crate::dist;
use crate::dist::Distribution;

fn ground_truth_prog(x1: f32, x2: f32) -> f32
{
    if x1 > x2
    {
        return 2.0 * x1 + x2;
    }

    return 2.0 / x2 - x1;
}

struct Holes
{
    par_1: dist::Categorical,
    par_2: dist::Normal,
    par_3: dist::Categorical,
    par_4: dist::Categorical,
    par_5: dist::Normal,
    par_6: dist::Categorical,
    par_7: dist::Categorical
}

fn hole_1(x1: f32, x2: f32, sign: f32) -> bool
{
    if sign == 0.0 {return x1 > x2;}
    if sign == 1.0 {return x1 < x2;}
    return x1 == x2;
}

fn hole_1_str(sign: f32) -> String
{
    if sign == 0.0 {return ">".to_owned();}
    if sign == 1.0 {return "<".to_owned();}
    return "==".to_owned();
}

fn hole_2(x1: f32, x2: f32, val: f32, sign_1: f32, sign_2: f32) -> f32
{
    let mut res: f32 = 0.0;

    if sign_1 == 0.0 {res += val + x1;}
    if sign_1 == 1.0 {res += val - x1;}
    if sign_1 == 2.0 {res += val * x1;}
    if sign_1 == 3.0 {res += val / x1;}

    if sign_2 == 0.0 {res += x2;}
    if sign_2 == 1.0 {res -= x2;}
    if sign_2 == 2.0 {res *= x2;}
    if sign_2 == 3.0 {res /= x2;}

    return res;
}

fn hole_2_str(val: f32, sign_1: f32, sign_2: f32) -> String
{
    let mut op_1 = "";
    let mut op_2 = "";

    if sign_1 == 0.0 {op_1 = "+";}
    if sign_1 == 1.0 {op_1 = "-";}
    if sign_1 == 2.0 {op_1 = "*";}
    if sign_1 == 3.0 {op_1 = "/";}

    if sign_2 == 0.0 {op_2 = "+";}
    if sign_2 == 1.0 {op_2 = "-";}
    if sign_2 == 2.0 {op_2 = "*";}
    if sign_2 == 3.0 {op_2 = "/";}

    return format!("{} {} x2 {} x1", val, op_1, op_2);
}

fn synth_prog(x1: f32, x2: f32, props: &Vec<f32>) -> f32
{
    if hole_1(x1, x2, props[0])
    {
        return hole_2(x1, x2, props[1], props[2], props[3]);
    }

    return hole_2(x1, x2, props[4], props[5], props[6])
}

fn square(x: f32) -> f32
{
    return x * x;
}

pub fn run_exp2(rate: f32) {
    let root = BitMapBackend::new("charts/complex.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Induction with Multiple Inputs", ("sans-serif", 15).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..20000f32, 0f32..10000f32).unwrap();

    chart
        .configure_mesh()
        .x_desc("Iterations")
        .y_desc("Loss")
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|x| format!("{:.0}", x))
        .draw().unwrap();

    let mut rng = rng::RNG::new(0);
    let test_inputs = vec![(5.8, 2.5), (5.0, 6.2), (7.4, 6.1), (5.5, 9.4)];
    let mut test_outputs = Vec::new();
    for t in 0..test_inputs.len()
    {
        let (a, b) = test_inputs[t];
        test_outputs.push(ground_truth_prog(a, b));
    }

    let num_mutations = 50;
    let num_iters = 20000;

    let par_1 = dist::Categorical::new(false, vec![0.0; 3]);
    let par_2 = dist::Normal::new(0.0, 1.0);
    let par_3 = dist::Categorical::new(false, vec![0.0; 4]);
    let par_4 = dist::Categorical::new(false, vec![0.0; 4]);
    let par_5 = dist::Normal::new(0.0, 1.0);
    let par_6 = dist::Categorical::new(false, vec![0.0; 4]);
    let par_7 = dist::Categorical::new(false, vec![0.0; 4]);

    let mut holes = Holes{par_1, par_2, par_3, par_4, par_5, par_6, par_7};
    let mut logger = Vec::new();

    for i in 0..num_iters
    {
        let mut trace_h1 = Vec::new();
        let mut trace_h2 = Vec::new();
        let mut trace_h3 = Vec::new();
        let mut trace_h4 = Vec::new();
        let mut trace_h5 = Vec::new();
        let mut trace_h6 = Vec::new();
        let mut trace_h7 = Vec::new();

        let mut objective = 0.0;
        let mut M = 0.0;
        let mut S = 0.0;

        for j in 0..num_mutations
        {
            let prop_1 = holes.par_1.sample(&mut rng) as f32;
            let prop_2 = holes.par_2.sample(&mut rng) as f32;
            let prop_3 = holes.par_3.sample(&mut rng) as f32;
            let prop_4 = holes.par_4.sample(&mut rng) as f32;
            let prop_5 = holes.par_5.sample(&mut rng) as f32;
            let prop_6 = holes.par_6.sample(&mut rng) as f32;
            let prop_7 = holes.par_7.sample(&mut rng) as f32;

            let props = vec![prop_1, prop_2, prop_3, prop_4, prop_5, prop_6, prop_7];

            let mut score = 0.0;
            for t in 0..test_inputs.len()
            {
                let (x1, x2) = test_inputs[t];
                score += square(synth_prog(x1, x2, &props) - test_outputs[t]);
            }

            score /= test_inputs.len() as f32;
            let old_M = M;
            M += (score - M) / (j as f32 + 1.0);
            S += (score - M) * (score - old_M);

            trace_h1.push((props[0] as usize, score));
            trace_h2.push((props[1], score));
            trace_h3.push((props[2] as usize, score));
            trace_h4.push((props[3] as usize, score));
            trace_h5.push((props[4], score));
            trace_h6.push((props[5] as usize, score));
            trace_h7.push((props[6] as usize, score));

            objective += score;
        }

        S = S / (num_mutations as f32 - 1.0);
        S = S.sqrt();

        for j in 0..num_mutations
        {
            let score = trace_h1[j].1;
            trace_h1[j].1 = (score - M) / S;

            let score = trace_h2[j].1;
            trace_h2[j].1 = (score - M) / S;

            let score = trace_h3[j].1;
            trace_h3[j].1 = (score - M) / S;
            
            let score = trace_h4[j].1;
            trace_h4[j].1 = (score - M) / S;

            let score = trace_h5[j].1;
            trace_h5[j].1 = (score - M) / S;

            let score = trace_h6[j].1;
            trace_h6[j].1 = (score - M) / S;

            let score = trace_h7[j].1;
            trace_h7[j].1 = (score - M) / S;          
        }

        objective /= num_mutations as f32;
        logger.push((i as f32, objective));

        let grad_h1 = holes.par_1.grad(trace_h1);
        let grad_h2 = holes.par_2.grad(trace_h2);
        let grad_h3 = holes.par_3.grad(trace_h3);
        let grad_h4 = holes.par_4.grad(trace_h4);
        let grad_h5 = holes.par_5.grad(trace_h5);
        let grad_h6 = holes.par_6.grad(trace_h6);
        let grad_h7 = holes.par_7.grad(trace_h7);

        holes.par_1.update(grad_h1, rate);
        holes.par_2.update(grad_h2, rate);
        holes.par_3.update(grad_h3, rate);
        holes.par_4.update(grad_h4, rate);
        holes.par_5.update(grad_h5, rate);
        holes.par_6.update(grad_h6, rate);
        holes.par_7.update(grad_h7, rate);
    }

    chart.draw_series(LineSeries::new(logger, &BLUE)).unwrap();
    root.present().unwrap();

    let mut synth_outputs = Vec::new();

    let prop_1 = holes.par_1.argmax() as f32;
    let prop_2 = holes.par_2.argmax() as f32;
    let prop_3 = holes.par_3.argmax() as f32;
    let prop_4 = holes.par_4.argmax() as f32;
    let prop_5 = holes.par_5.argmax() as f32;
    let prop_6 = holes.par_6.argmax() as f32;
    let prop_7 = holes.par_7.argmax() as f32;

    let props = vec![prop_1, prop_2, prop_3, prop_4, prop_5, prop_6, prop_7];
    let mut score = 0.0;
    
    for t in 0..test_inputs.len()
    {
        let (x1, x2) = test_inputs[t];
        let output = synth_prog(x1, x2, &props);
        score += square(output - test_outputs[t]);
        synth_outputs.push(output);
    }
    score /= test_inputs.len() as f32;

    println!("Ground truth outputs: {:?}", test_outputs);
    println!("Induction outputs: {:?}", synth_outputs);
    println!("Learning rate: {}", rate);
    println!("MSE Loss: {}", score);

    let op_1 = hole_1_str(prop_1);
    let op_2 = hole_2_str(prop_2, prop_3, prop_4);
    let op_3 = hole_2_str(prop_5, prop_6, prop_7);

    let prog = format!(r#"
    fn synth_prog(x1: f32, x2: f32, props: &Vec<f32>) -> f32
    {{
        if x1 {op_1} x2
        {{
            return {op_2};
        }}

        return {op_3};
    }}"#);

    let gt = r#"
    fn ground_truth_prog(x1: f32, x2: f32) -> f32
    {
        if x1 > x2
        {
            return 2.0 * x1 + x2;
        }

        return 2.0 / x2 - x1;
    }"#;

    println!("{}", prog);
    println!("{}", gt);
}