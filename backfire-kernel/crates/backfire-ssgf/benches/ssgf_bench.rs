// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel SSGF Benchmarks
// ─────────────────────────────────────────────────────────────────────
//! Criterion benchmarks for the SSGF geometry engine.
//!
//! Covers every hot-path component:
//!   - Decoders (gram_softplus, rbf)
//!   - Micro-cycle engine (single step, full microcycle)
//!   - Spectral bridge (Laplacian + Jacobi eigensolver + gauge)
//!   - Cost terms (individual + composite U_total)
//!   - Finite-difference gradient
//!   - Full outer-cycle step
//!   - Audio mapping
//!
//! All timings must stay well under the 50 ms deadline
//! (Backfire Prevention Protocols §2.2).

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use backfire_physics::params::{build_knm_matrix, OMEGA_N};
use backfire_ssgf::costs::{
    compute_costs, cost_c10_boundary, cost_c8_phase, cost_micro, regularise_graph, CostWeights,
};
use backfire_ssgf::decoder::{
    decode_gram_softplus, decode_rbf, gradient_analytic_gram, gradient_fd,
};
use backfire_ssgf::engine::{SSGFConfig, SSGFEngine};
use backfire_ssgf::micro::MicroCycleEngine;
use backfire_ssgf::spectral::{GaugeMethod, SpectralBridge};
use backfire_ssgf::GradientMethod;

const N: usize = 16;

// ── Helpers ───────────────────────────────────────────────────────────

fn make_omega() -> Vec<f64> {
    OMEGA_N.to_vec()
}

fn make_knm_flat() -> Vec<f64> {
    build_knm_matrix()
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect()
}

fn make_z() -> Vec<f64> {
    (0..N * (N - 1) / 2)
        .map(|i| (i as f64 * 0.07).sin() * 0.5)
        .collect()
}

fn make_theta() -> Vec<f64> {
    (0..N)
        .map(|i| (i as f64 * 0.4).sin() * std::f64::consts::PI)
        .collect()
}

fn make_w() -> Vec<f64> {
    let z = make_z();
    let mut w = vec![0.0; N * N];
    decode_gram_softplus(&z, N, &mut w);
    w
}

fn make_engine() -> SSGFEngine {
    SSGFEngine::new(&make_omega(), &make_knm_flat(), SSGFConfig::default())
}

// ── Decoder benchmarks ───────────────────────────────────────────────

fn bench_decode_gram_softplus(c: &mut Criterion) {
    let z = make_z();
    let mut w = vec![0.0; N * N];
    c.bench_function("decode_gram_softplus_16x16", |b| {
        b.iter(|| decode_gram_softplus(black_box(&z), N, &mut w))
    });
}

fn bench_decode_rbf(c: &mut Criterion) {
    // For RBF, z is N*D dimensional (use D=8)
    let z: Vec<f64> = (0..N * 8).map(|i| (i as f64 * 0.03).sin()).collect();
    let mut w = vec![0.0; N * N];
    c.bench_function("decode_rbf_16x16_d8", |b| {
        b.iter(|| decode_rbf(black_box(&z), N, 1.0, &mut w))
    });
}

// ── Micro-cycle benchmarks ───────────────────────────────────────────

fn bench_micro_single_step(c: &mut Criterion) {
    let omega = make_omega();
    let k = make_knm_flat();
    let mut engine = MicroCycleEngine::new(N, &omega, &k, 0.001, 0.1, 0.02, 0.05, 0.1, 42);
    let mut theta = make_theta();
    let w = make_w();

    c.bench_function("micro_single_step_16", |b| {
        b.iter(|| engine.step(black_box(&mut theta), black_box(&w), None))
    });
}

fn bench_micro_10_steps(c: &mut Criterion) {
    let omega = make_omega();
    let k = make_knm_flat();
    let mut engine = MicroCycleEngine::new(N, &omega, &k, 0.001, 0.1, 0.02, 0.05, 0.1, 42);
    let mut theta = make_theta();
    let w = make_w();

    c.bench_function("micro_10_steps_16", |b| {
        b.iter(|| engine.run_microcycle(black_box(&mut theta), black_box(&w), 10, None))
    });
}

fn bench_micro_with_pgbo(c: &mut Criterion) {
    let omega = make_omega();
    let k = make_knm_flat();
    let mut engine = MicroCycleEngine::new(N, &omega, &k, 0.001, 0.1, 0.02, 0.05, 0.1, 42);
    let mut theta = make_theta();
    let w = make_w();
    let h_munu = vec![0.01; N * N]; // Dummy PGBO tensor

    c.bench_function("micro_10_steps_16_pgbo", |b| {
        b.iter(|| {
            engine.run_microcycle(
                black_box(&mut theta),
                black_box(&w),
                10,
                Some(black_box(&h_munu)),
            )
        })
    });
}

fn bench_order_parameter(c: &mut Criterion) {
    let omega = make_omega();
    let k = make_knm_flat();
    let engine = MicroCycleEngine::new(N, &omega, &k, 0.001, 0.1, 0.0, 0.0, 0.0, 42);
    let theta = make_theta();

    c.bench_function("order_parameter_16", |b| {
        b.iter(|| engine.compute_order_parameter(black_box(&theta)))
    });
}

// ── Spectral benchmarks ──────────────────────────────────────────────

fn bench_spectral_eigenpairs(c: &mut Criterion) {
    let w = make_w();
    let mut bridge = SpectralBridge::new(N, GaugeMethod::Sign);
    let mut eigvals = vec![0.0; N];
    let mut eigvecs = vec![0.0; N * N];

    c.bench_function("spectral_eigenpairs_16x16", |b| {
        b.iter(|| {
            bridge.compute_eigenpairs(black_box(&w), &mut eigvals, &mut eigvecs);
        })
    });
}

fn bench_spectral_laplacian(c: &mut Criterion) {
    let w = make_w();
    let mut bridge = SpectralBridge::new(N, GaugeMethod::None);

    c.bench_function("spectral_laplacian_16x16", |b| {
        b.iter(|| {
            bridge.normalised_laplacian(black_box(&w));
        })
    });
}

// ── Cost benchmarks ──────────────────────────────────────────────────

fn bench_cost_micro(c: &mut Criterion) {
    let theta = make_theta();
    c.bench_function("cost_micro_16", |b| {
        b.iter(|| cost_micro(black_box(&theta)))
    });
}

fn bench_cost_c8_phase(c: &mut Criterion) {
    let theta = make_theta();
    let eigvals = vec![
        0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
    ];
    c.bench_function("cost_c8_phase_16", |b| {
        b.iter(|| cost_c8_phase(black_box(&theta), black_box(&eigvals)))
    });
}

fn bench_cost_c10_boundary(c: &mut Criterion) {
    let w = make_w();
    let mask = vec![
        true, true, true, true, true, true, true, true, false, false, false, false, false, false,
        false, false,
    ];
    c.bench_function("cost_c10_boundary_16", |b| {
        b.iter(|| cost_c10_boundary(black_box(&w), N, black_box(&mask)))
    });
}

fn bench_regularise_graph(c: &mut Criterion) {
    let w = make_w();
    let eigvals = vec![
        0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
    ];
    c.bench_function("regularise_graph_16", |b| {
        b.iter(|| regularise_graph(black_box(&w), black_box(&eigvals), 0.01, 0.1))
    });
}

fn bench_compute_costs_all(c: &mut Criterion) {
    let theta = make_theta();
    let w = make_w();
    let mut bridge = SpectralBridge::new(N, GaugeMethod::Sign);
    let mut eigvals = vec![0.0; N];
    let mut eigvecs = vec![0.0; N * N];
    bridge.compute_eigenpairs(&w, &mut eigvals, &mut eigvecs);
    let phi_target = vec![0.0; N * N];
    let mask = vec![false; N];
    let weights = CostWeights::default();

    c.bench_function("compute_costs_all_16", |b| {
        b.iter(|| {
            compute_costs(
                black_box(&theta),
                black_box(&w),
                N,
                black_box(&eigvals),
                black_box(&eigvecs),
                black_box(&phi_target),
                black_box(&mask),
                black_box(&weights),
                0.5,
                None,
            )
        })
    });
}

// ── Gradient benchmark ───────────────────────────────────────────────

fn bench_gradient_fd(c: &mut Criterion) {
    let z = make_z();
    let mut w_scratch = vec![0.0; N * N];
    let theta = make_theta();
    let phi_target = vec![0.0; N * N];
    let mask = vec![false; N];
    let weights = CostWeights::default();

    c.bench_function("gradient_fd_120dim", |b| {
        b.iter(|| {
            let theta = theta.clone();
            let phi_target = phi_target.clone();
            let mask = mask.clone();
            let weights = weights.clone();
            let mut spectral = SpectralBridge::new(N, GaugeMethod::None);
            let mut grad_eigvals = vec![0.0; N];
            let mut grad_eigvecs = vec![0.0; N * N];

            let decode_fn = |z: &[f64], n: usize, w: &mut [f64]| {
                decode_gram_softplus(z, n, w);
            };
            let mut loss_fn = |w: &[f64]| -> f64 {
                spectral.compute_eigenpairs(w, &mut grad_eigvals, &mut grad_eigvecs);
                compute_costs(
                    &theta,
                    w,
                    N,
                    &grad_eigvals,
                    &grad_eigvecs,
                    &phi_target,
                    &mask,
                    &weights,
                    0.5,
                    None,
                )
                .u_total
            };

            gradient_fd(
                black_box(&z),
                N,
                &mut w_scratch,
                &decode_fn,
                &mut loss_fn,
                1e-5,
            )
        })
    });
}

// ── Engine benchmarks ────────────────────────────────────────────────

fn bench_engine_outer_step(c: &mut Criterion) {
    let mut engine = make_engine();
    c.bench_function("engine_outer_step", |b| b.iter(|| engine.outer_step()));
}

fn bench_engine_5_steps(c: &mut Criterion) {
    let mut engine = make_engine();
    c.bench_function("engine_5_outer_steps", |b| b.iter(|| engine.run(5)));
}

fn bench_engine_audio_mapping(c: &mut Criterion) {
    let mut engine = make_engine();
    engine.outer_step();
    c.bench_function("engine_audio_mapping", |b| {
        b.iter(|| engine.audio_mapping())
    });
}

fn bench_engine_init(c: &mut Criterion) {
    let omega = make_omega();
    let k = make_knm_flat();
    c.bench_function("engine_init_16", |b| {
        b.iter(|| SSGFEngine::new(black_box(&omega), black_box(&k), SSGFConfig::default()))
    });
}

// ── Deadline compliance: full pipeline ───────────────────────────────

fn bench_deadline_single_outer_step(c: &mut Criterion) {
    let mut engine = make_engine();
    // Warm up (first step is cold-start)
    engine.outer_step();

    c.bench_function("DEADLINE_outer_step_warmed", |b| {
        b.iter(|| engine.outer_step())
    });
}

// ── Analytic gradient benchmark ──────────────────────────────────────

fn bench_gradient_analytic_gram(c: &mut Criterion) {
    let z = make_z();
    let w = make_w();
    let mask = vec![false; N];
    let weights = CostWeights::default();
    let mut du_dw = vec![0.0; N * N];

    c.bench_function("gradient_analytic_gram_120dim", |b| {
        b.iter(|| {
            gradient_analytic_gram(
                black_box(&z),
                N,
                black_box(&w),
                black_box(&mask),
                black_box(&weights),
                &mut du_dw,
            )
        })
    });
}

fn bench_engine_outer_step_analytic(c: &mut Criterion) {
    let omega = make_omega();
    let k = make_knm_flat();
    let cfg = SSGFConfig {
        gradient_method: GradientMethod::Analytic,
        ..SSGFConfig::default()
    };
    let mut engine = SSGFEngine::new(&omega, &k, cfg);
    c.bench_function("engine_outer_step_analytic", |b| {
        b.iter(|| engine.outer_step())
    });
}

fn bench_deadline_analytic_warmed(c: &mut Criterion) {
    let omega = make_omega();
    let k = make_knm_flat();
    let cfg = SSGFConfig {
        gradient_method: GradientMethod::Analytic,
        ..SSGFConfig::default()
    };
    let mut engine = SSGFEngine::new(&omega, &k, cfg);
    // Warm up
    engine.outer_step();

    c.bench_function("DEADLINE_outer_step_analytic_warmed", |b| {
        b.iter(|| engine.outer_step())
    });
}

// ── Groups ───────────────────────────────────────────────────────────

criterion_group!(decoders, bench_decode_gram_softplus, bench_decode_rbf,);

criterion_group!(
    micro,
    bench_micro_single_step,
    bench_micro_10_steps,
    bench_micro_with_pgbo,
    bench_order_parameter,
);

criterion_group!(
    spectral,
    bench_spectral_laplacian,
    bench_spectral_eigenpairs,
);

criterion_group!(
    costs,
    bench_cost_micro,
    bench_cost_c8_phase,
    bench_cost_c10_boundary,
    bench_regularise_graph,
    bench_compute_costs_all,
);

criterion_group!(gradient, bench_gradient_fd, bench_gradient_analytic_gram,);

criterion_group!(
    engine,
    bench_engine_init,
    bench_engine_outer_step,
    bench_engine_outer_step_analytic,
    bench_engine_5_steps,
    bench_engine_audio_mapping,
    bench_deadline_single_outer_step,
    bench_deadline_analytic_warmed,
);

criterion_main!(decoders, micro, spectral, costs, gradient, engine);
