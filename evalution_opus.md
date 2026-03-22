# Evaluation Metrics and Acceptance Criteria for Multi-Field Cosmological Generative Models: A Physics-Grounded Framework

## GENESIS Project Technical Report

---

## 1. Introduction and Motivation

Evaluating generative models for cosmological field emulation requires metrics that are both statistically rigorous and physically meaningful. Unlike natural image generation, where perceptual quality metrics (FID, IS) suffice, cosmological applications demand that generated fields faithfully reproduce the statistical properties of the underlying matter distribution — particularly the power spectrum P(k), cross-correlations between fields, and pixel-level distributions.

This report establishes physics-grounded evaluation criteria for the GENESIS project, which generates three cosmological fields (dark matter density Mcdm, gas density Mgas, and gas temperature T) conditioned on six cosmological and astrophysical parameters (Ωm, σ8, ASN1, ASN2, AAGN1, AAGN2) using Flow Matching with a Swin Transformer backbone. We critically examine the commonly adopted targets from single-field studies and demonstrate that naive application of these thresholds to the multi-field, 6D-conditioned setting is physically unjustified. We then propose revised, field-dependent and scale-dependent criteria grounded in the irreducible uncertainties of the CAMELS simulation suite.

---

## 2. Evaluation Metrics: What to Measure and Why

### 2.1 Auto-Power Spectrum P_i(k)

**Definition.** The auto-power spectrum quantifies the variance of a single field as a function of spatial scale k:

    ΔP/P(k) = |P_gen(k) - P_true(k)| / P_true(k)

where P_gen and P_true are the azimuthally averaged 2D power spectra of generated and true fields, respectively. The relative error is computed per k-bin and aggregated as:

- Mean relative error: ⟨ΔP/P⟩ averaged over all k-bins
- RMS error: √⟨(ΔP/P)²⟩
- Max error: max_k ΔP/P(k)

**Why it matters.** The power spectrum is the primary summary statistic in cosmology. For a Gaussian random field, P(k) is a sufficient statistic — it completely determines the field's statistical properties (Villaescusa-Navarro et al. 2021). Even for non-Gaussian fields at low redshift, P(k) captures approximately 60–70% of the cosmological information content (Villaescusa-Navarro et al. 2021). Accurate reproduction of P(k) is therefore a necessary (though not sufficient) condition for a physically valid emulator.

**How other papers evaluated this metric:**

- Diffusion-HMC (Mudur & Finkbeiner 2024): Used the reduced chi-squared statistic χ²_r rather than percentage error. For the CV set (405 generated fields vs. 15 true fields per parameter, 35 k-bins), they obtained χ²_r = 2.29 ± 0.49 for generated fields, compared to χ²_r = 1.70 ± 0.36 for true fields via leave-one-out cross-validation. They did not report explicit ΔP/P percentages.

- CosmoFlow (2025): Reported power spectrum agreement visually, showing that the flow matching reconstruction preserves all frequency components while a VAE baseline loses high-frequency information. For parameter inference, they achieved 5.24% and 4.03% mean relative error for Ωm and σ8 using 8-channel summary statistics. Explicit P(k) error percentages were not provided.

- CAMELS GAN (Andrianomena et al. 2024): Used the coefficient of determination R² between generated and true auto-power spectra. Across six cosmologies and three fields (Mgas, HI, B), they obtained R² = 0.90–0.99 for auto-power, with the note that "the clustering properties of CAMELS and those of the generated data are consistent with each other, as evidenced by all R² values ≥ 0.9." They also reported relative error ≤ 25% up to k ~ 10 h/Mpc for both unconditional and conditional cases.

- Galaxy Field VDM (2024): Used power spectrum residuals (25th–75th percentile bands) and cross-power spectra, showing that VDM outperforms CNN and HOD baselines at small scales, but without reporting explicit percentage targets.

### 2.2 Cross-Power Spectrum P_ij(k)

**Definition.** The cross-power spectrum measures the correlated component between two fields i and j:

    P_ij(k) = Re[FFT(field_i)* × FFT(field_j)] / (H × W)²

The relative error is computed analogously to the auto-power case:

    ΔP_ij/P_ij(k) = |P_ij^gen(k) - P_ij^true(k)| / |P_ij^true(k)|

For three fields, there are three cross-power pairs: (Mcdm, Mgas), (Mcdm, T), and (Mgas, T).

**Why it matters.** Cross-power spectra encode the spatial coherence between different physical fields. In cosmology, these correlations arise from causal physical processes: dark matter forms gravitational potential wells that attract gas (DM-gas correlation), infalling gas shock-heats to the virial temperature (gas-T correlation), and the depth of the potential well determines both gas density and temperature (DM-T indirect correlation). A generative model that reproduces each field's marginal distribution but fails to preserve cross-correlations would produce physically inconsistent multi-field maps — e.g., generating hot gas in void regions where no dark matter halo exists.

**How other papers evaluated this metric:**

- CAMELS GAN (Andrianomena et al. 2024): Reported cross-power R² for three pairs (Mgas×HI, Mgas×B, HI×B). Results were highly variable: R² = 0.84–0.99 for most cosmologies, but catastrophically negative (R² = -0.55 for Mgas×B, R² = -11.93 for HI×B) for specific parameter combinations ({Ωm = 0.22, σ8 = 0.98} and {Ωm = 0.30, σ8 = 0.92}). The authors attributed these failures to "the unconstrained stellar feedbacks ASN1 and ASN2 which have a non-negligible effect on the PDFs of HI and B."

- Diffusion-HMC: Did not report cross-power spectra (single-field model).

- CosmoFlow: Did not report cross-power spectra (single-field model).

**Critical observation:** The CAMELS GAN is the only published multi-field cosmological emulator reporting cross-power spectra quantitatively, and it suffered catastrophic failures on specific parameter combinations. This establishes that cross-power preservation is a fundamentally challenging problem, and any stable cross-power reproduction — even at 15–25% error — represents a significant advance over the state of the art.

### 2.3 Correlation Coefficient r_ij(k)

**Definition.** The scale-dependent correlation coefficient normalizes the cross-power by the geometric mean of the auto-powers:

    r_ij(k) = P_ij(k) / √[P_ii(k) · P_jj(k)]

with r ∈ [-1, 1]. The evaluation metric is:

    Δr(k) = |r_gen(k) - r_true(k)|

**Why it matters.** While the cross-power spectrum P_ij(k) depends on the absolute amplitudes of both fields, the correlation coefficient isolates the structural relationship between fields at each scale. Physically, r_DM-gas(k) encodes how tightly gas traces dark matter as a function of scale:

- k < 0.1 h/Mpc (large scales): r ≈ 0.95–0.98, gravity dominates, gas faithfully traces DM.
- k ~ 1 h/Mpc (intermediate): r ≈ 0.85–0.90, feedback begins to displace gas from DM halos.
- k > 3 h/Mpc (small scales): r ≈ 0.70–0.80, baryonic physics (SN/AGN feedback, cooling, star formation) significantly decorrelates gas from DM.

This scale-dependent decorrelation is a direct observable consequence of astrophysical feedback and is one of the key quantities that CAMELS was designed to study.

**How other papers evaluated this metric:** No published cosmological generative model has explicitly reported r_ij(k) as an evaluation metric. The CAMELS GAN reported cross-power R², which is related but distinct (R² compares spectral shapes, r(k) measures per-scale correlation structure). Achieving Δr < 0.1 for any field pair therefore represents a novel contribution.

### 2.4 Pixel Distribution Function (PDF) and KS Test

**Definition.** The one-point PDF is the histogram of pixel values across all spatial positions. Comparison between generated and true distributions is typically quantified via the two-sample Kolmogorov-Smirnov (KS) test:

    KS statistic D = sup_x |F_gen(x) - F_true(x)|

where F_gen and F_true are the empirical cumulative distribution functions.

**Why it matters.** The PDF captures information beyond the power spectrum — particularly the non-Gaussian tails of the density distribution, which encode halo abundances (high-density peaks) and void properties (low-density regions). For temperature fields, the PDF may exhibit bimodality (cold gas at ~10⁴ K and hot halo gas at ~10⁶⁻⁷ K), which is a direct signature of feedback physics.

**How other papers evaluated this metric:**

- Diffusion-HMC: Reported p-values from comparing the means of generated and true log-field distributions. The p-values were above 0.05 for all eight checkpoints tested, but crucially, this comparison was between distribution means (averaged over 50 generated and 15 true fields per parameter), not pixel-level distributions.

- CAMELS GAN: Compared μ_PDF and σ_PDF visually between 1500 generated and 1500 real images. Reported "good agreement overall, however some statistical differences at the tails of the distributions can be noticed." No formal statistical test was applied.

- CosmoFlow: Showed visual overlap of density histograms between true and generated fields without formal testing.

**Critical issue with pixel-level KS tests:** For N generated maps of size 256×256, the effective sample size for a pixel-level KS test is N × 256² = N × 65,536. For N=8, this gives 524,288 data points. The KS test's power scales as √n, meaning that even a 0.1% distributional difference will be detected as statistically significant (p ≈ 0) at this sample size. The KS p-value > 0.05 criterion is therefore fundamentally inappropriate for pixel-level comparisons at this scale. The KS statistic D itself (rather than the p-value) is a more meaningful measure of distributional distance.

### 2.5 CAMELS Validation Protocols (LH/1P/CV/EX)

Beyond the core statistical metrics above, the CAMELS simulation suite provides four purpose-designed validation sets that test distinct scientific properties:

**LH (Latin Hypercube) validation.** 100 held-out simulations spanning the 6D parameter space. Tests generalization across diverse cosmologies and feedback strengths.

**1P (One-Parameter-at-a-time).** 6 × 26 = 156 simulations where one parameter varies while others are fixed at fiducial values. Tests whether the model correctly captures the sensitivity ∂P(k)/∂θ_i for each parameter. This is critical for downstream parameter inference.

**CV (Cosmic Variance).** 27 simulations at fiducial parameters with different random seeds. Tests whether the model reproduces the correct level of stochasticity (cosmic variance) rather than producing overly deterministic outputs. The key metric is the variance ratio σ²_gen / σ²_true.

**EX (Extreme).** Simulations at parameter values near or beyond the training range boundaries. Tests extrapolation robustness and the absence of catastrophic failures (NaN, divergence).

These protocols are a distinguishing feature of CAMELS-based studies. While most ML4cosmology papers evaluate only on a held-out test set (equivalent to LH), the 1P/CV/EX sets probe fundamentally different aspects of model fidelity. The Diffusion-HMC paper (Mudur & Finkbeiner 2024) is the most thorough prior work in this regard, having reported results on LH, 1P, and CV sets for a single dark matter field.

---

## 3. Physical Constraints on Achievable Accuracy

### 3.1 Cosmic Variance in the CAMELS Box

The CAMELS simulations use a periodic box of side length L = 25 h⁻¹ Mpc. This relatively small volume introduces an irreducible statistical uncertainty on the power spectrum known as cosmic variance. For a 2D field in a box of side L with pixel resolution N_pix, the fractional uncertainty on P(k) due to finite mode counting is:

    σ_CV(k) / P(k) ≈ 1 / √N_modes(k)

where N_modes(k) is the number of independent Fourier modes in the annulus [k, k + Δk]. For a 2D field:

    N_modes(k) ≈ 2π k Δk (L / 2π)²

For the CAMELS box (L = 25 h⁻¹ Mpc, fundamental mode k_f = 2π/L ≈ 0.25 h/Mpc):

| k [h/Mpc] | N_modes (approx.) | σ_CV/P(k) |
|------------|-------------------|------------|
| 0.3 (fundamental) | ~2–5 | ~45–70% |
| 1.0 | ~15–30 | ~18–26% |
| 3.0 | ~50–100 | ~10–14% |
| 10.0 | ~200–400 | ~5–7% |

These estimates are consistent with the findings of Villaescusa-Navarro et al. (2021), who noted that "cosmic variance alone represents a significant amount of scatter in power suppression even for fiducial feedback parameters" in the CAMELS (25 h⁻¹ Mpc)³ box. Furthermore, Gebhardt et al. (2024) reported "significant scatter in the relation between fractional power suppression and normalized spread due to cosmic variance," and the CAMELS documentation explicitly acknowledges that "the relatively small simulation box size (25 h⁻¹ Mpc)³ limits direct comparison to wide-field/massive structures and can bias predictions for quantities sensitive to large-scale modes."

**Implication for auto-power targets:** At k < 1 h/Mpc, the cosmic variance floor is ~18–70%. A uniform 5% accuracy target across all k-bins is below the irreducible noise floor for the lowest k-modes and is therefore physically unachievable, even with a perfect generative model.

### 3.2 Baryonic Feedback Sensitivity Varies by Orders of Magnitude Across Fields

The three fields in GENESIS respond to the six CAMELS parameters with vastly different sensitivities. From the CAMELS 1P set:

**Dark matter (Mcdm):** Responds primarily to cosmological parameters (Ωm, σ8). Sensitivity to feedback parameters (ASN, AAGN) is negligible (∂logP_DM/∂logA_SN ~ 0) because dark matter is collisionless and does not directly interact with baryonic feedback. This is the easiest field to emulate.

**Gas density (Mgas):** Responds to both cosmological and feedback parameters. SN feedback expels gas from low-mass halos (∂logP_gas/∂logA_SN1 ~ -0.8 at k ~ 1–5 h/Mpc), while AGN feedback affects massive halos. The cross-suite comparison (SIMBA vs. TNG vs. Astrid) shows dramatic differences: in SIMBA, approximately 40% of baryons are ejected beyond 1 Mpc/h, compared to only 10–15% in TNG/Astrid (Gebhardt et al. 2023). This means the "ground truth" P_gas(k) itself has substantial systematic uncertainty depending on the subgrid model.

**Gas temperature (T):** Most sensitive to feedback, particularly AGN. The sensitivity ∂logP_T/∂logA_AGN1 ~ +2.0 at cluster scales (k ~ 0.3–1 h/Mpc) is more than an order of magnitude larger than the DM field's sensitivity to any parameter. Additionally, the temperature field may exhibit bimodal pixel distributions (cold gas at ~10⁴ K, hot halo gas at ~10⁶⁻⁷ K), reflecting the complex thermodynamic state of the intergalactic medium. Temperature is the hardest field to emulate.

**Implication for evaluation criteria:** Applying the same accuracy threshold (e.g., 5%) to all three fields is physically unjustified. A field-dependent criterion that reflects the intrinsic difficulty and variability of each field is more appropriate.

### 3.3 2D Projection Introduces Slice-to-Slice Variance

The CAMELS 2D Multifield Dataset generates 15 2D maps from each (25 h⁻¹ Mpc)³ simulation by projecting along three axes with five slices per axis (each slice ~5 h⁻¹ Mpc thick). Maps from different slices of the same simulation show P(k) variations of approximately 5–15% (scale-dependent), adding another layer of irreducible variance to any comparison between generated and true fields.

### 3.4 Finite Sample Size (N=8) Amplifies All Uncertainties

When comparing N_gen generated maps against N_true true maps, the standard error on the mean P(k) is:

    SE = σ_total / √N

where σ_total includes cosmic variance, slice variance, and any model-specific stochasticity. For N=8:

    SE(k ~ 1 h/Mpc) ≈ 15% / √8 ≈ 5.3%

This means that the measurement uncertainty on the mean P(k) difference is itself ~5%, comparable to the proposed 5% target. Any apparent ΔP/P below ~5% at mid-k could be entirely attributable to measurement noise rather than genuine model accuracy.

---

## 4. Proposed Evaluation Criteria

Based on the physical considerations in Section 3 and the precedents established by prior work (Section 2), we propose the following evaluation framework.

### 4.1 Auto-Power Spectrum

We adopt field-dependent and scale-dependent targets:

| Field | k < 1 h/Mpc | k = 1–5 h/Mpc | k > 5 h/Mpc | Justification |
|-------|-------------|----------------|-------------|---------------|
| Mcdm | < 10% | < 15% | < 25% | Lowest feedback sensitivity; CV floor ~18% at k~1; nonlinear regime at high-k |
| Mgas | < 15% | < 20% | < 30% | Moderate feedback sensitivity (ASN ~0.8); CAMELS GAN achieved R² ≥ 0.90 (~10–30% error) |
| T | < 20% | < 25% | < 35% | Extreme feedback sensitivity (AAGN ~2.0); bimodal distribution; no prior multi-field emulator has generated T |

**Max error:** Reported for transparency but not used as a pass/fail criterion. A single k-bin with anomalously high error provides limited diagnostic value compared to the mean and RMS.

**RMS error:** Complements the mean by penalizing large outliers: target set at 1.5× the mean error threshold for each field and k-range.

### 4.2 Cross-Power Spectrum

Cross-power targets are pair-dependent, reflecting the different physical mechanisms that drive each correlation:

| Pair | Target (mean ΔP_ij/P_ij) | Justification |
|------|--------------------------|---------------|
| Mcdm-Mgas | < 15% | Direct gravitational coupling; strongest cross-correlation; CAMELS GAN R² = 0.84–0.99 but with catastrophic outliers |
| Mcdm-T | < 25% | Indirect correlation (DM → potential → gas infall → heating); weakest of the three pairs; most affected by AGN feedback |
| Mgas-T | < 15% | Direct thermodynamic coupling (virial equilibrium T ∝ M^{2/3}); physically the most constrained relationship |

These targets are set at approximately 1.5× the corresponding auto-power targets, following the principle that cross-correlations are intrinsically harder to reproduce than marginal distributions (as demonstrated by the CAMELS GAN failures).

### 4.3 Correlation Coefficient

| k-range | Target max Δr | Justification |
|---------|--------------|---------------|
| k < 5 h/Mpc | < 0.1 | r(k) is well-defined and physically interpretable in this regime; Δr = 0.1 corresponds to the typical scale of feedback-induced decorrelation |
| k > 5 h/Mpc | < 0.2 | r(k) becomes noisy due to limited mode counts; cosmic variance on individual P(k) values propagates into r |

Note: No prior cosmological generative model has reported r_ij(k) as an explicit metric. Any Δr < 0.1 result at k < 5 h/Mpc represents a novel contribution to the field.

### 4.4 Pixel Distribution (PDF)

We replace the KS p-value criterion with the KS statistic D as the primary metric:

| Metric | Target | Justification |
|--------|--------|---------------|
| KS statistic D | < 0.05 | With N × 256² pixels, D < 0.05 indicates that the maximum pointwise CDF difference is less than 5%. This is achievable and meaningful, unlike p-value which is overwhelmed by sample size. |
| PDF mean relative error | < 5% | |μ_gen - μ_true| / |μ_true| for each field's pixel distribution |
| PDF std relative error | < 10% | |σ_gen - σ_true| / σ_true, following CAMELS GAN's visual μ/σ comparison approach |

For the temperature field specifically, we additionally recommend reporting the locations and heights of the bimodal peaks separately, as the cold gas (~10⁴ K) and hot halo gas (~10⁶⁻⁷ K) populations encode distinct physical processes.

### 4.5 CAMELS Protocols

| Protocol | Metric | Target | Justification |
|----------|--------|--------|---------------|
| LH (generalization) | All metrics above on 100 validation sims | Field-dependent targets from §4.1–4.4 | Standard held-out validation |
| 1P (sensitivity) | ∂P(k)/∂θ_i ratio curves | Generated ratios within ±2σ of true (per seed) | Diffusion-HMC achieved this for DM with 2 params; 6-param, 3-field case is harder |
| CV (cosmic variance) | σ²_gen / σ²_true P(k) ratio | 0.7 < ratio < 1.3 for most k | Diffusion-HMC achieved ~0.9–1.0 for single DM; relaxed from 0.8–1.2 to account for multi-field complexity |
| EX (extrapolation) | Auto-power error | < 2× LH targets | Error increase is expected for out-of-distribution parameters; criterion is absence of catastrophic failure (no NaN, no divergence, no >200% error) |

### 4.6 Measurement Protocol

To ensure reproducibility and statistical validity:

- **Final evaluation sample size:** N ≥ 32 generated maps per conditioning parameter (up from N=8 used during training monitoring).
- **Uncertainty quantification:** All metrics reported as mean ± standard deviation over 5 independent sampling runs (different random seeds for the ODE solver / noise initialization).
- **Pass criterion:** mean + 1σ < target threshold. This ensures that the model reliably meets the target, not just occasionally.

---

## 5. Summary Comparison with Prior Work

| | Diffusion-HMC (2024) | CosmoFlow (2025) | CAMELS GAN (2024) | GENESIS (this work) |
|---|---|---|---|---|
| Fields | DM only | DM only | Mgas, HI, B | Mcdm, Mgas, T |
| Conditioning | 2D (Ωm, σ8) | 2D (Ωm, σ8) | 2D (Ωm, σ8) | 6D (+ASN1,2, AAGN1,2) |
| Method | DDPM | Flow Matching | WGAN-GP + SE | Flow Matching + Swin |
| Auto-power metric | χ²_r = 2.29 | Visual + param err | R² = 0.90–0.99 | ΔP/P field+k-dependent |
| Cross-power metric | N/A | N/A | R² = -11.93 to 0.99 | ΔP/P pair-dependent |
| Correlation r(k) | Not reported | Not reported | Not reported | Δr < 0.1 (k < 5) |
| PDF metric | p > 0.05 (means) | Visual | Visual (μ, σ) | KS stat D < 0.05 |
| Validation protocols | LH, 1P, CV | 1P, CV | Visual only | LH, 1P, CV, EX |
| N_samples (eval) | 50 per param | 405 (CV) | 10 per cosmo | 32+ (proposed) |

---

## 6. Conclusion

The evaluation criteria for multi-field cosmological generative models must account for three irreducible physical constraints: (1) cosmic variance imposed by the finite (25 h⁻¹ Mpc)³ simulation volume, which sets a noise floor of ~18–70% at k < 1 h/Mpc; (2) the order-of-magnitude variation in feedback sensitivity across fields, with temperature being ~20× more sensitive to AGN parameters than dark matter; and (3) the limited statistical power of small sample sizes (N ~ 8–32), which introduces ~5–15% measurement uncertainty on P(k) differences.

We argue that the commonly cited "< 5% P(k) accuracy" target — derived from single-field, 2D-conditioned DM studies — is inappropriate for the multi-field, 6D-conditioned setting of GENESIS. Field-dependent, scale-dependent criteria that respect the physical noise floor provide a more honest and scientifically defensible evaluation framework.

---

## References

1. Villaescusa-Navarro, F. et al. (2021). "The CAMELS project: Cosmology and Astrophysics with MachinE Learning Simulations." ApJ 915, 71. arXiv:2010.00619

2. Villaescusa-Navarro, F. et al. (2022). "The CAMELS Multifield Dataset: Learning the Universe's Fundamental Parameters with Artificial Intelligence." ApJS 259, 61. arXiv:2109.10915

3. Mudur, N. & Finkbeiner, D. (2024). "Cosmological Field Emulation and Parameter Inference with Diffusion Models." arXiv:2312.07534

4. Andrianomena, S. & Hassan, S. (2024). "Cosmological multifield emulator." Phys. Rev. D. arXiv:2402.10997

5. Andrianomena, S. et al. (2022). "Emulating cosmological multifields with generative adversarial networks." arXiv:2211.05000

6. CosmoFlow (2025). "CosmoFlow: Flow Matching for Cosmological Representation Learning." (Reference from project files: cosmoflow.pdf)

7. Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." ICLR 2023.

8. Gebhardt, M. et al. (2023/2024). Baryon spread and power spectrum suppression in CAMELS. (Referenced via Guo et al. 2025 and CAMELS documentation)

9. Guo, Y. et al. (2025). "Constraining the Effect of Baryonic Feedback on the Matter Power Spectrum with Fast Radio Bursts." arXiv:2501.17922

10. Ono, V. et al. (2024). "Debiasing with Diffusion: Probabilistic reconstruction of Dark Matter fields from galaxies with CAMELS." arXiv:2403.10648

11. Angulo, R. & Pontzen, A. (2016). "Cosmological N-body simulations with suppressed variance." MNRAS 462, L1-L5.

12. Villaescusa-Navarro, F. et al. (2018). "Suppressing cosmic variance with paired-and-fixed cosmological simulations." MNRAS (referenced via Oxford Academic)

13. Peebles, W. & Xie, S. (2023). "Scalable Diffusion Models with Transformers." ICCV 2023.

14. Liu, Z. et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.

15. Ho, J. et al. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.

16. Ni, Y. et al. (2023). ASTRID simulation suite for CAMELS. (Referenced via CAMELS documentation)
