"""
rebuild_pattern_tables.py
=========================
Standalone script — no notebook variables needed.
Parses CRE_clean.abc directly to rebuild rhythm pattern_tables in
cre_calibration.pkl from all 1195 corpus tunes.

Run from anywhere:
    python rebuild_pattern_tables.py

Or from Jupyter (any notebook):
    %run /path/to/rebuild_pattern_tables.py
"""

import re
import pickle
import os

# ── Paths ─────────────────────────────────────────────────────────────────────

ABC_FILE  = '/Users/maz/Desktop/PhD Galway/folk_ngram_analysis-master/cre_corpus/abc/CRE_clean.abc'
PKL_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cre_calibration.pkl')

# ── Rhythm helpers (identical to pto_seqplusplus_new.ipynb) ───────────────────

_BAR_INFO = {
    '4/4':  (4.0,  16),
    'C':    (4.0,  16),
    'C|':   (4.0,  16),
    '3/4':  (3.0,  12),
    '2/4':  (2.0,   8),
    '6/8':  (3.0,  12),
    '9/8':  (4.5,  18),
    '12/8': (6.0,  24),
}

_TS_NORMALISE = {'C': '4/4', 'C|': '4/4'}


def _compute_histogram(onsets_qn, time_sig):
    """Normalised onset histogram within one bar (onsets in quarter notes)."""
    bar_length, bins = _BAR_INFO.get(time_sig, (4.0, 16))
    histogram = [0.0] * bins
    for t in onsets_qn:
        bar_pos = t % bar_length
        histogram[int((bar_pos / bar_length) * bins) % bins] += 1
    total = sum(histogram)
    if total > 0:
        histogram = [h / total for h in histogram]
    return tuple(histogram)


# ── ABC parser ────────────────────────────────────────────────────────────────

def _default_unit(time_sig):
    """Default L: unit (in quarter notes) when L: is absent in an ABC tune."""
    # ABC standard: if M numerator/denominator >= 0.75 → L=1/8; else L=1/16
    ts = time_sig.strip()
    if '/' in ts:
        num, den = ts.split('/')
        ratio = int(num) / int(den)
    else:
        ratio = 1.0  # C or C|
    return 0.5 if ratio >= 0.75 else 0.25   # 1/8 or 1/16 of a whole note in QN


def _parse_note_body(text, unit_qn):
    """
    Walk through the note body of an ABC tune and yield onset times (in QN).
    Handles: note/rest durations, slashes, triplets/tuplets, broken rhythm (><),
    ties, grace notes, chords.
    """
    i = 0
    n = len(text)
    current_time = 0.0

    # Stack for tuplet scaling: (factor, remaining_notes)
    tuplet = None  # (scale_factor, notes_remaining)
    broken_next = 1.0  # multiplier for next note after > or <

    while i < n:
        c = text[i]

        # ── skip whitespace & bar-related chars ───────────────────────────
        if c in ' \t\r\n|:':
            i += 1
            continue

        # ── grace notes {A} or {/A} ───────────────────────────────────────
        if c == '{':
            while i < n and text[i] != '}':
                i += 1
            i += 1
            continue

        # ── decoration / annotation ──────────────────────────────────────
        if c in '!+':
            while i < n and text[i] not in '!+ ':
                i += 1
            i += 1
            continue

        # ── chord [CEG] — treat as single note with duration of first note ─
        if c == '[':
            # check it's not [| or [: (barline variants)
            if i + 1 < n and text[i+1] not in '|:':
                i += 1   # skip [
                # skip until ]
                chord_start = i
                while i < n and text[i] != ']':
                    i += 1
                chord_text = text[chord_start:i]
                i += 1  # skip ]
                # parse duration after ] if any
                dur_num, dur_den, i = _read_duration(text, i, n)
                dur_qn = unit_qn * dur_num / dur_den * broken_next
                # apply tuplet
                if tuplet:
                    dur_qn *= tuplet[0]
                    tuplet = (tuplet[0], tuplet[1] - 1) if tuplet[1] > 1 else None
                # emit first note of chord as onset
                if chord_text and chord_text[0] not in 'zx':
                    yield current_time
                current_time += dur_qn
                broken_next = 1.0
                continue
            else:
                i += 1
                continue

        # ── tuplets (3 / (2 / etc. ──────────────────────────────────────
        if c == '(':
            if i + 1 < n and text[i+1].isdigit():
                p = int(text[i+1])
                # optional :q:r form — skip it
                i += 2
                if i < n and text[i] == ':':
                    i += 1
                    while i < n and text[i].isdigit(): i += 1
                    if i < n and text[i] == ':':
                        i += 1
                        while i < n and text[i].isdigit(): i += 1
                # p notes in the time of q (q defaults to 2 for p=3,6, else p-1 or similar)
                q = {2: 3, 3: 2, 4: 3, 5: 4, 6: 4, 7: 6, 8: 6, 9: 6}.get(p, 2)
                scale = q / p
                tuplet = (scale, p)
            else:
                i += 1
            continue

        # ── broken rhythm > and < ─────────────────────────────────────────
        if c in '><':
            # The PREVIOUS note was already emitted; adjust current_time
            # > means prev note gets 3/2, next gets 1/2 of their value
            # We cannot change the already-emitted onset, but we can scale
            # the time offset we already advanced.  Approximate by scaling
            # the next note duration only.
            broken_next = 0.5 if c == '>' else 1.5
            i += 1
            continue

        # ── accidentals ──────────────────────────────────────────────────
        if c in '^_=':
            i += 1
            if i < n and text[i] in '^_':  # double sharp/flat
                i += 1
            continue

        # ── note or rest ─────────────────────────────────────────────────
        if c in 'ABCDEFGabcdefgzxZ':
            is_rest = c in 'zxZ'
            i += 1
            # octave marks
            while i < n and text[i] in "',":
                i += 1
            # duration
            dur_num, dur_den, i = _read_duration(text, i, n)
            dur_qn = unit_qn * dur_num / dur_den * broken_next
            broken_next = 1.0
            # tie/slur — just skip the hyphen; doesn't affect onset counting
            if i < n and text[i] == '-':
                i += 1
            # apply tuplet
            if tuplet:
                dur_qn *= tuplet[0]
                tuplet = (tuplet[0], tuplet[1] - 1) if tuplet[1] > 1 else None
            if not is_rest:
                yield current_time
            current_time += dur_qn
            continue

        # ── skip everything else (repeat signs, section labels, etc.) ────
        i += 1


def _read_duration(text, i, n):
    """Read optional [num][/[den]] duration modifier starting at position i."""
    dur_num = 1
    dur_den = 1
    if i < n and text[i].isdigit():
        dur_num = 0
        while i < n and text[i].isdigit():
            dur_num = dur_num * 10 + int(text[i])
            i += 1
    if i < n and text[i] == '/':
        i += 1
        if i < n and text[i].isdigit():
            dur_den = 0
            while i < n and text[i].isdigit():
                dur_den = dur_den * 10 + int(text[i])
                i += 1
        else:
            # bare / means half the current numerator
            dur_den = 2
    return dur_num, dur_den, i


def parse_abc_file(path):
    """
    Yield (title, canonical_ts, histogram) for every tune in the ABC file.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    tunes = re.split(r'\n%%%\n', content)

    for raw in tunes:
        raw = raw.strip()
        if not raw:
            continue

        # ── extract headers ───────────────────────────────────────────────
        title_m = re.search(r'^T:(.+)', raw, re.MULTILINE)
        ts_m    = re.search(r'^M:(.+)', raw, re.MULTILINE)
        unit_m  = re.search(r'^L:\s*(\d+)/(\d+)', raw, re.MULTILINE)

        if not title_m or not ts_m:
            continue

        title = title_m.group(1).strip()
        ts    = ts_m.group(1).strip()

        if unit_m:
            unit_qn = int(unit_m.group(1)) * 4.0 / int(unit_m.group(2))
        else:
            unit_qn = _default_unit(ts)

        canonical_ts = _TS_NORMALISE.get(ts, ts)
        if canonical_ts not in _BAR_INFO:
            continue  # unsupported time sig — skip

        # ── extract note body (everything after K: line) ──────────────────
        k_match = re.search(r'^K:.*$', raw, re.MULTILINE)
        if not k_match:
            continue
        note_body = raw[k_match.end():]

        # ── collect onsets ────────────────────────────────────────────────
        onsets = list(_parse_note_body(note_body, unit_qn))
        if not onsets:
            continue

        hist = _compute_histogram(onsets, canonical_ts)
        yield title, canonical_ts, hist


# ── Main ──────────────────────────────────────────────────────────────────────

def rebuild(abc_file=ABC_FILE, pkl_file=PKL_FILE):
    print(f"Parsing: {abc_file}")
    print(f"Target:  {pkl_file}")

    pattern_tables = {}
    tune_counts    = {}
    processed = 0
    skipped   = 0

    for title, ts, hist in parse_abc_file(abc_file):
        if ts not in pattern_tables:
            pattern_tables[ts] = {}
            tune_counts[ts]    = 0
        pattern_tables[ts][hist] = pattern_tables[ts].get(hist, 0) + 1
        tune_counts[ts] += 1
        processed += 1

    total = sum(tune_counts.values())
    print(f"\nProcessed {processed} tunes (skipped {skipped}):")
    for ts in sorted(pattern_tables.keys()):
        n_pat = len(pattern_tables[ts])
        n_tun = tune_counts[ts]
        print(f"  {ts}: {n_pat} unique patterns from {n_tun} tunes")
    print(f"  Total: {total}")

    with open(pkl_file, 'rb') as f:
        calibration = pickle.load(f)

    calibration['pattern_tables'] = pattern_tables
    calibration['total_corpus']   = total

    with open(pkl_file, 'wb') as f:
        pickle.dump(calibration, f)

    print(f"\nSaved. NPC mean={calibration['npc_mean']:.2f}, "
          f"NPC values={len(calibration.get('npc_values', []))}")


if __name__ == '__main__':
    rebuild()
