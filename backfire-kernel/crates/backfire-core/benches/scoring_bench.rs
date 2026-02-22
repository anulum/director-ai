// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel Scoring Benchmarks
// ─────────────────────────────────────────────────────────────────────
//! Criterion benchmarks proving the hot path completes within the
//! 50 ms deadline specified in Backfire Prevention Protocols §2.2.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use backfire_core::{
    CoherenceScorer, HeuristicNli, InMemoryKnowledge, SafetyKernel, StreamingKernel,
};
use backfire_types::BackfireConfig;

// ── CoherenceScorer.review() ────────────────────────────────────────

fn bench_scorer_review(c: &mut Criterion) {
    let scorer = CoherenceScorer::new(
        BackfireConfig::default(),
        Arc::new(HeuristicNli),
        Arc::new(InMemoryKnowledge::new()),
    );
    c.bench_function("scorer_review", |b| {
        b.iter(|| {
            scorer.review(
                black_box("What color is the sky?"),
                black_box("The sky is blue, consistent with reality"),
            )
        })
    });
}

// ── SafetyKernel.stream_output() ────────────────────────────────────

fn bench_safety_kernel_10_tokens(c: &mut Criterion) {
    let kernel = SafetyKernel::new(0.5);
    let tokens: Vec<&str> = (0..10).map(|_| "token ").collect();
    c.bench_function("safety_kernel_10tok", |b| {
        b.iter(|| kernel.stream_output(black_box(&tokens), &|_| 0.8))
    });
}

fn bench_safety_kernel_100_tokens(c: &mut Criterion) {
    let kernel = SafetyKernel::new(0.5);
    let tokens: Vec<&str> = (0..100).map(|_| "token ").collect();
    c.bench_function("safety_kernel_100tok", |b| {
        b.iter(|| kernel.stream_output(black_box(&tokens), &|_| 0.8))
    });
}

// ── StreamingKernel.stream_tokens() ─────────────────────────────────

fn bench_streaming_kernel_10_tokens(c: &mut Criterion) {
    let config = BackfireConfig::default();
    let kernel = StreamingKernel::new(config);
    let tokens: Vec<&str> = (0..10).map(|_| "token ").collect();
    c.bench_function("streaming_kernel_10tok", |b| {
        b.iter(|| kernel.stream_tokens(black_box(&tokens), &|_| 0.8))
    });
}

fn bench_streaming_kernel_100_tokens(c: &mut Criterion) {
    let config = BackfireConfig::default();
    let kernel = StreamingKernel::new(config);
    let tokens: Vec<&str> = (0..100).map(|_| "token ").collect();
    c.bench_function("streaming_kernel_100tok", |b| {
        b.iter(|| kernel.stream_tokens(black_box(&tokens), &|_| 0.8))
    });
}

fn bench_streaming_kernel_1000_tokens(c: &mut Criterion) {
    let config = BackfireConfig::default();
    let kernel = StreamingKernel::new(config);
    let tokens: Vec<&str> = (0..1000).map(|_| "token ").collect();
    c.bench_function("streaming_kernel_1000tok", |b| {
        b.iter(|| kernel.stream_tokens(black_box(&tokens), &|_| 0.8))
    });
}

// ── Full pipeline: score + gate ─────────────────────────────────────

fn bench_full_pipeline(c: &mut Criterion) {
    let scorer = Arc::new(CoherenceScorer::new(
        BackfireConfig::default(),
        Arc::new(HeuristicNli),
        Arc::new(InMemoryKnowledge::new()),
    ));
    let kernel = SafetyKernel::new(0.5);
    let tokens: Vec<&str> = (0..10).map(|_| "token ").collect();

    c.bench_function("full_pipeline_10tok", |b| {
        let scorer = scorer.clone();
        b.iter(|| {
            // Score
            let (approved, _score) = scorer.review(
                black_box("What color is the sky?"),
                black_box("The sky is blue"),
            );
            // Gate
            if approved {
                kernel.stream_output(black_box(&tokens), &|_| 0.8);
            }
        })
    });
}

criterion_group!(
    benches,
    bench_scorer_review,
    bench_safety_kernel_10_tokens,
    bench_safety_kernel_100_tokens,
    bench_streaming_kernel_10_tokens,
    bench_streaming_kernel_100_tokens,
    bench_streaming_kernel_1000_tokens,
    bench_full_pipeline,
);
criterion_main!(benches);
