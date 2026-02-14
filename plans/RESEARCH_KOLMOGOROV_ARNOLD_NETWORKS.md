# Research: Kolmogorov-Arnold Networks (KAN)

**Status:** Research
**Created:** 2025-12-29
**Last Updated:** 2025-12-29

## Overview

Kolmogorov-Arnold Networks (KANs) are a novel neural network architecture based on the **Kolmogorov-Arnold representation theorem**. They represent a fundamental departure from traditional MLPs and could significantly impact neuroevolution.

## The Kolmogorov-Arnold Representation Theorem

The theorem states that any multivariate continuous function can be represented as:

```
f(x₁, x₂, ..., xₙ) = Σᵢ₌₀²ⁿ Φᵢ(Σⱼ₌₁ⁿ φᵢⱼ(xⱼ))
```

Where:
- `φᵢⱼ` are univariate functions (inner functions)
- `Φᵢ` are univariate functions (outer functions)
- The sum uses only **2n+1** terms

**Key insight:** Any function of n variables can be decomposed into compositions and sums of univariate functions only.

## KAN vs MLP Architecture

| Aspect | MLP | KAN |
|--------|-----|-----|
| **Activation** | Fixed on nodes (ReLU, tanh) | Learnable on edges |
| **Weights** | Scalar values | Univariate spline functions |
| **Parameters** | W matrices | Spline coefficients per edge |
| **Expressivity** | Universal approximator | Faster scaling laws |
| **Interpretability** | Black box | Visualizable, interactive |

```
MLP Layer:              KAN Layer:
    y = σ(Wx + b)           y = Σ φᵢ(xᵢ)

    x₁ ──┐                  x₁ ──φ₁₁──┐
    x₂ ──┼──[σ]──> y        x₂ ──φ₂₁──┼──> y
    x₃ ──┘                  x₃ ──φ₃₁──┘

  (fixed σ, learned W)    (learned φᵢⱼ splines)
```

## Spline Parameterization

Each edge function φᵢⱼ is a **B-spline**:

```
φᵢⱼ(x) = Σₖ cₖ Bₖ(x)

Where:
- Bₖ(x) are B-spline basis functions
- cₖ are learnable coefficients
- Grid points define spline resolution
```

**Benefits:**
- Local support (efficient updates)
- Smooth derivatives (stable training)
- Grid refinement (adaptive precision)

## Computational Considerations

| Factor | Impact |
|--------|--------|
| **Parameters** | More per edge, but fewer edges needed |
| **Memory** | Spline grids require storage |
| **Inference** | Spline evaluation is O(k) per edge |
| **Training** | Gradient through spline coefficients |

## Relevance to Neuroevolution

### Potential Benefits

1. **Smaller Networks:** KANs achieve same accuracy with fewer layers
2. **Interpretability:** Can visualize evolved functions
3. **Scientific Discovery:** Agents might evolve interpretable behaviors
4. **Efficiency:** Fewer parameters to evolve

### Implementation Considerations

1. **Genome Representation:**
   ```erlang
   %% Current TWEANN genome
   {neuron, Id, ActivationFn, Weights}

   %% KAN genome would be
   {kan_edge, FromId, ToId, SplineCoeffs, GridPoints}
   ```

2. **Mutation Operators:**
   - Perturb spline coefficients
   - Refine/coarsen grid
   - Add/remove edges
   - Split/merge splines

3. **Crossover:**
   - Align spline grids
   - Blend coefficients
   - Exchange edge functions

### Research Questions

1. **Can KANs be evolved effectively?**
   - Spline mutations vs weight mutations
   - Grid adaptation during evolution

2. **Do evolved KANs remain interpretable?**
   - Post-evolution analysis
   - Function extraction from agents

3. **Performance in RL tasks?**
   - Sense-Think-Act with KAN brains
   - Temporal dynamics with LTC-KAN hybrids

## KAN Variants (2024-2025)

| Variant | Key Innovation | Application |
|---------|----------------|-------------|
| **Wav-KAN** | Wavelet basis functions | Signal processing |
| **Fourier-KAN** | Fourier series edges | Periodic functions |
| **CKAN** | Convolutional structure | Image/sequence data |
| **KA-GNN** | Graph neural networks | Molecular property prediction |
| **LKAN** | Linear approximation | Genomic sequences |

## Next Steps

1. [ ] Implement basic KAN layer in Erlang
2. [ ] Create KAN-compatible genome representation
3. [ ] Design mutation operators for splines
4. [ ] Compare evolved KAN vs TWEANN on benchmarks
5. [ ] Explore LTC-KAN hybrid for temporal tasks

## Potential LTC-KAN Integration

```
                    ┌─────────────────────────────┐
                    │      LTC-KAN Neuron         │
                    ├─────────────────────────────┤
                    │  τ(t) · dx/dt = -x + φ(u)  │
                    │                             │
                    │  Where φ is a learned       │
                    │  spline function (KAN)      │
                    │  instead of fixed tanh      │
                    └─────────────────────────────┘
```

This could combine:
- LTC's temporal dynamics and stability
- KAN's expressivity and interpretability
- Neuroevolution's optimization power

## References

- [KAN: Kolmogorov-Arnold Networks (ICLR 2025)](https://arxiv.org/abs/2404.19756)
- [OpenReview Discussion](https://openreview.net/forum?id=Ozo7qJ5vZi)
- [Wav-KAN with Wavelets](https://www.semanticscholar.org/paper/KAN:-Kolmogorov-Arnold-Networks-Liu-Wang/14fdab35cc6288083a38a92392af3f1da0b95a90)
- [KAGNNs for Graph Learning](https://arxiv.org/abs/2406.18380)
- [Kolmogorov-Arnold Networks for Molecular Properties](https://www.nature.com/articles/s42256-025-01087-7)
- [Convolutional KAN for Intrusion Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC11733237/)
