# MusicPTO — Evolutionary Irish Music Generation with PTO

Evolutionary music generation using [Program Trace Optimisation (PTO)](https://github.com/Program-Trace-Optimisation/PTO). The system evolves Irish-style melodies by optimising the random decisions made by a hierarchical music generator (Seq++) against a corpus-calibrated fitness function.

---

## Overview

The GA operates on **program traces** — the sequence of random decisions (`rnd.*` calls) made during melody generation. Crossover and mutation act on these traces directly; replaying them through the generator produces new melodies. This means the search space is the space of programs that generate music, not the music itself.

Two generators are compared:

| Generator | Description |
|---|---|
| **Seq++** | Hierarchical: generates base chunks, reuses them via a grammar with HO/AU pattern transformations |
| **Flat** | Baseline: same pitch/duration space as Seq++, no structural reuse — melody is a flat sequence of independent notes |

The only intended difference between the two is **structural reuse**. Everything else (time signature, feel, duration types, weights) is matched.

---

## Fitness Function

Three components, weighted and normalised against the Irish tune corpus:

| Component | Weight | Method |
|---|---|---|
| **Rhythm** | 0.4 | KNN cosine similarity to corpus onset histograms (k=5) |
| **Length** | 0.3 | KDE score against corpus encoding-length distribution |
| **Complexity** | 0.3 | KDE score on NPC residuals (length-adjusted) |

NPC (Nested Pattern Complexity) measures the cost of the hierarchical encoding of a tune. The complexity score rewards tunes whose NPC is typical for their encoding length in the corpus.

---

## Key Files

| File | Description |
|---|---|
| `pto_seqplusplus_new.ipynb` | Main notebook — generator, fitness, evolution, analysis |
| `cre_calibration.pkl` | Corpus calibration data (NPC values, rhythm patterns, length distribution) |
| `irish_npc_results.csv` | Per-tune NPC and encoding-length data from the Irish corpus |
| `abc_lookup.json` | ABC notation lookup for corpus tunes |
| `decoder.py` | Decodes grammar/chunk structures to flat melodies; saves MIDI |
| `calculate_irish_npc_complete.py` | Computes NPC for the full Irish corpus |
| `rebuild_pattern_tables.py` | Rebuilds rhythm pattern tables from corpus |

---

## Setup

```bash
pip install git+https://github.com/Program-Trace-Optimisation/PTO.git
pip install numpy scipy matplotlib pandas midiutil
```

Open and run `pto_seqplusplus_new.ipynb` top to bottom. Cells are numbered and sectioned.

---

## Notebook Sections

1. **Setup** — install PTO, load calibration
2. **Corpus Analysis** — NPC distribution, rhythm profiles, pattern frequency
3. **Generator** — Seq++ hierarchical generator
4. **Fitness** — complexity (NPC-KDE), length (KDE), rhythm (KNN)
5. **Evolution** — PTO GA on Seq++ and flat generator; 20-run comparison
6. **Analysis & Export** — plots, MIDI export, statistical comparison
