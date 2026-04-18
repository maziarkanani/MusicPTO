## Fitness Function Design and Generator Calibration for Seq++ with PTO

### Overview

This section describes the design and calibration of the fitness function and generator parameters for evolving Irish traditional music using Program Trace Optimization (PTO) with the Seq++ hierarchical music representation. The fitness function has two components — structural complexity and rhythmic similarity — both calibrated against the CRE (Cree) collection.

### Generator Calibration

The Seq++ generator produces hierarchical music structures consisting of chunks (beat-level melodic fragments), higher-order (HO) patterns, anti-unification (AU) patterns, variation functions (transposition, retrograde, inversion, augmentation, diminution, pitch change, rhythm change), and a grammar that defines the overall musical form.

Since PTO operates on the space of random decisions made by the generator, the generator's parameter ranges directly determine what the evolutionary search can explore. To ensure the generated tunes occupy the same structural space as the corpus, we analysed the corpus to extract statistics on each structural element.

A scanning function was written to process all corpus tunes through the encoder and variation detection pipeline, extracting counts of HO patterns, AU patterns, and each variation function type. The key findings were:

- **HO patterns**: Rare in the corpus (mean=0.13 per tune, only 11% of tunes have any). Generator `ho_patterns_range` was set to (0, 1).
- **AU patterns**: Common (mean=1.77, median=2, 75% of tunes have at least one). Generator `au_patterns_range` was set to (0, 3), up from the initial (0, 0).
- **Variation functions**: Transposition dominates at 80% of all detected variations, with retrograde, inversion, augmentation, and diminution each contributing small percentages. The generator's transform selection was weighted accordingly (80% T, with the remainder split among G, I, P, H) rather than using uniform selection.
- **Transform probability**: 41% of encoding symbols in the corpus are variation functions rather than plain chunk references. Generator `transform_prob` was raised from 0.15 to 0.40.

The generator's `chunks_range` and `chunk_length_range` were also adjusted empirically, testing multiple configurations and measuring the resulting NPC distributions against the corpus target. The final parameters produce tunes whose NPC values overlap well with the corpus range.

### Fitness Component 1: Structural Complexity (NPC)

#### Length-Adjusted Scoring

Initial experiments revealed that NPC correlates strongly with melody length (R² = 0.657). The corpus NPC distribution is also right-skewed (mean=137.78, median=125, range 54–480), making a symmetric Gaussian reward inappropriate. A naive Gaussian centered on the corpus mean would penalise short simple tunes (which are actually common in the corpus) too harshly.

Two adjustments were made:

1. **Length normalization via regression residuals.** A quadratic polynomial was fitted to the corpus data relating encoding length (in beats) to NPC: `predicted_NPC = 0.0006 × length² + 1.26 × length + 56.6`. The residual (actual NPC minus predicted NPC) captures how complex a tune is *relative to its length*, with a mean of 0 and standard deviation of 28.7.

2. **Kernel Density Estimation (KDE) on residuals.** Rather than assuming a Gaussian distribution, a KDE was fitted to the corpus residuals. The complexity score for a generated tune is computed as `KDE(residual) / max(KDE)`, yielding a value in [0, 1] where 1.0 corresponds to the most common residual value (i.e., the generated tune's complexity is exactly what would be expected for a corpus tune of the same length).

This approach means a short tune with low NPC and a long tune with high NPC can both receive high complexity scores, as long as their NPC is typical for their respective lengths. The KDE captures the actual shape of the residual distribution without parametric assumptions.

### Fitness Component 2: Rhythmic Similarity (KNN)

Each tune's note onsets are collapsed into a single-bar histogram. The corpus provides a table of all unique onset histograms grouped by time signature.

Initial analysis of the corpus rhythm patterns revealed that almost every tune has a unique rhythm profile — for example, 4/4 tunes have 580 unique patterns across 583 tunes. This makes pattern frequency (commonness) a meaningless metric, as nearly all patterns occur exactly once.

The rhythm scoring was therefore redesigned as a K-Nearest Neighbours (KNN) approach: cosine similarity is computed between the generated tune's histogram and every corpus pattern in the same time signature, and the mean similarity across the K=5 nearest neighbours is taken as the rhythm score. This naturally rewards tunes that sit in dense regions of the corpus rhythm space — a tune whose rhythm is similar to many corpus tunes scores higher than one that resembles only a single outlier pattern. The KNN approach captures both closeness and density in one metric without requiring an explicit commonness measure.

Time signatures are handled separately throughout: a generated 6/8 tune is only compared to 6/8 corpus patterns, a 4/4 tune to 4/4 patterns, and so on.

### Combined Fitness

The two components are combined as a weighted sum:

```
fitness = (w_complexity × complexity_score + w_rhythm × rhythm_score) / (w_complexity + w_rhythm)
```

Both weights are currently set to 0.5, giving equal importance to structural complexity and rhythmic similarity. A melody length filter (40–120 notes) is applied, assigning a fitness of 0 to tunes outside this range to focus the evolutionary search on musically reasonable lengths.