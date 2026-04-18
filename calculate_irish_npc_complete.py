"""
Complete NPC Calculator for Irish Tune Dataset

This script includes:
1. All variation detection functions (repetition, transposition, etc.)
2. The incremental encoder
3. NPC calculator with our latest rules (separate pitch/duration counting)
4. Analysis code with detailed statistics

Usage in notebook:
    %run calculate_irish_npc_complete.py
    
    # Then run:
    df = run_npc_analysis(chunks_list, use_degree=True)
    print_statistics(df)
"""

import re
import statistics as stats
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Try to import sksequitur
try:
    from sksequitur import parse, Production
    HAS_SEQUITUR = True
except ImportError:
    print("Warning: sksequitur not found. Grammar complexity will be 0.")
    HAS_SEQUITUR = False


# =============================================================================
# VARIATION DETECTION FUNCTIONS
# =============================================================================

def repetition(motive, melody, format_type=None):
    """Detect exact repetitions of motif in melody."""
    repetitions = []
    motive_len = len(motive)
    melody_len = len(melody)

    for i in range(melody_len - motive_len + 1):
        segment = melody[i:i + motive_len]
        if segment == motive:
            repetitions.append(i)

    return repetitions


def transposition(motive, melody, format_type=None):
    """Detect transpositions (same duration pattern, shifted pitch)."""
    def get_pitch(note):
        return note[0] if isinstance(note, tuple) else note

    transpositions = []
    L = len(motive)
    M = len(melody)

    for i in range(M - L + 1):
        segment = melody[i:i + L]

        # Compute semitone interval between first notes
        interval = get_pitch(segment[0]) - get_pitch(motive[0])

        # Skip exact repetition (interval 0 handled elsewhere)
        if interval == 0:
            continue

        # Require both pitch AND duration match
        same_duration = all(segment[j][1] == motive[j][1] for j in range(L))
        if not same_duration:
            continue

        # Verify consistent transposition for all notes
        if all(get_pitch(segment[j]) - get_pitch(motive[j]) == interval for j in range(L)):
            transpositions.append({
                "position": i,
                "interval": interval
            })

    return transpositions


def retrograde(motive, melody, format_type=None):
    """Detect retrograde (reversed pitch AND duration)."""
    def get_pitch(note):
        return note[0] if isinstance(note, tuple) else note

    retrogrades = []
    L = len(motive)
    M = len(melody)

    for i in range(M - L + 1):
        segment = melody[i:i + L]

        ok = True
        for j in range(L):
            # Both pitch AND duration should be from the reversed position
            if get_pitch(segment[j]) != get_pitch(motive[L - 1 - j]):
                ok = False
                break
            if segment[j][1] != motive[L - 1 - j][1]:
                ok = False
                break

        if ok:
            retrogrades.append(i)

    return retrogrades


def inversion(motive, melody, format_type=None):
    """Detect true inversions (pitches mirrored around first note axis)."""
    def get_pitch(note):
        return note[0] if isinstance(note, tuple) else note

    inversions = []
    L = len(motive)
    M = len(melody)

    # Axis of inversion = first pitch of motif
    axis = get_pitch(motive[0])

    for i in range(M - L + 1):
        segment = melody[i:i + L]

        # First note must match the same axis pitch
        if get_pitch(segment[0]) != axis:
            continue

        # Durations must match exactly
        if not all(segment[j][1] == motive[j][1] for j in range(L)):
            continue

        # For every note: (pitch distance from axis) must be mirrored
        ok = True
        for j in range(L):
            motif_pitch = get_pitch(motive[j])
            expected_inv = axis - (motif_pitch - axis)
            if get_pitch(segment[j]) != expected_inv:
                ok = False
                break

        if ok:
            inversions.append(i)

    return inversions


def augmentation(motive, melody, format_type=None):
    """Detect augmentation (same pitches, scaled durations > 1)."""
    if not isinstance(motive[0], tuple):
        return []

    augmentations = []
    motive_len = len(motive)
    melody_len = len(melody)

    for i in range(melody_len - motive_len + 1):
        segment = melody[i:i + motive_len]

        # Pitch pattern must match
        if all(segment[j][0] == motive[j][0] for j in range(motive_len)):
            # Compute duration ratio
            rhythm_factor = segment[0][1] / motive[0][1]

            if rhythm_factor <= 1:  # skip diminutions
                continue

            # Verify consistent scaling across notes
            if all(abs(segment[j][1] - motive[j][1] * rhythm_factor) <= 1 for j in range(motive_len)):
                augmentations.append({
                    "position": i,
                    "rhythm_factor": rhythm_factor
                })

    return augmentations


def diminution(motive, melody, format_type=None):
    """Detect diminution (same pitches, scaled durations < 1)."""
    if not isinstance(motive[0], tuple):
        return []

    diminutions = []
    motive_len = len(motive)
    melody_len = len(melody)

    for i in range(melody_len - motive_len + 1):
        segment = melody[i:i + motive_len]

        # Pitch pattern must match
        if all(segment[j][0] == motive[j][0] for j in range(motive_len)):
            # Compute duration scaling factor
            rhythm_factor = segment[0][1] / motive[0][1]

            if rhythm_factor >= 1:  # skip augmentations
                continue

            # Verify consistent scaling
            if all(abs(segment[j][1] - motive[j][1] * rhythm_factor) <= 1 for j in range(motive_len)):
                diminutions.append({
                    "position": i,
                    "rhythm_factor": rhythm_factor
                })

    return diminutions


def rhythm_change(motive, melody, format_type=None):
    """Detect rhythm changes (same pitches, few duration changes)."""
    if not isinstance(motive[0], tuple):
        return []

    results = []
    L = len(motive)
    M = len(melody)
    max_changes = L // 4

    for i in range(M - L + 1):
        segment = melody[i:i + L]

        # Pitch pattern must match exactly
        if all(segment[j][0] == motive[j][0] for j in range(L)):
            # Record duration changes
            changed = [
                {"note_index": j, "new_value": segment[j][1]}
                for j in range(L)
                if segment[j][1] != motive[j][1]
            ]

            if 0 < len(changed) <= max_changes:
                results.append({
                    "position": i,
                    "changed": changed
                })

    return results


def pitch_change(motive, melody, format_type=None):
    """Detect pitch changes (same durations, few pitch changes)."""
    def get_pitch(note):
        return note[0] if isinstance(note, tuple) else note

    results = []
    L = len(motive)
    M = len(melody)
    max_changes = L // 4

    for i in range(M - L + 1):
        segment = melody[i:i + L]

        # Durations must match exactly
        if not all(segment[j][1] == motive[j][1] for j in range(L)):
            continue

        # Find pitch differences
        changed = [
            {"note_index": j, "new_value": get_pitch(segment[j])}
            for j in range(L)
            if get_pitch(segment[j]) != get_pitch(motive[j])
        ]

        if 0 < len(changed) <= max_changes:
            results.append({
                "position": i,
                "changed": changed
            })

    return results


# Default variation functions dictionary
VAR_FUNCS = {
    "repetition": repetition,
    "transposition": transposition,
    "inversion": inversion,
    "retrograde": retrograde,
    "augmentation": augmentation,
    "diminution": diminution,
    "rhythm_change": rhythm_change,
    "pitch_change": pitch_change,
}


# =============================================================================
# INCREMENTAL ENCODER
# =============================================================================

def make_symbol(cname, vtype, params):
    """Generate symbolic representation of a variation."""
    if vtype == "repetition":
        return f"{cname}"
    elif vtype == "transposition":
        return f"T({cname},{params.get('interval',0)})"
    elif vtype == "inversion":
        return f"I({cname})"
    elif vtype == "retrograde":
        return f"G({cname})"
    elif vtype == "augmentation":
        return f"A({cname},{params.get('rhythm_factor',1.0):.2f})"
    elif vtype == "diminution":
        return f"D({cname},{params.get('rhythm_factor',1.0):.2f})"
    elif vtype == "rhythm_change":
        changed = params.get("changed")
        if changed:
            changes_str = ",".join(f"{c['note_index']},{c['new_value']}" for c in changed)
            return f"H({cname},{changes_str})"
        return f"H({cname})"
    elif vtype == "pitch_change":
        changed = params.get("changed")
        if changed:
            changes_str = ",".join(f"{c['note_index']},{c['new_value']}" for c in changed)
            return f"P({cname},{changes_str})"
        return f"P({cname})"
    else:
        return f"{vtype[:1].upper()}({cname})"


def encode_chunks_incremental(chunks, var_funcs=None):
    """
    Incremental parser with nested variation support.
    
    For each chunk, finds the best matching variation by:
    1. Checking chunks in order they were added (oldest first)
    2. Taking the FIRST valid match found
    3. Building a library of both unique chunks and their meaningful variations
    
    Returns:
        encoding: List of symbolic tokens
        known_chunks: Dict mapping symbolic names to chunk data
        unique_chunks: Dict of base chunks only (c_1, c_2, etc.)
    """
    if var_funcs is None:
        var_funcs = VAR_FUNCS
    
    known_chunks = {}      # Maps symbolic names to chunk data
    chunk_order = []       # Tracks insertion order
    encoding = []          # Output sequence of symbolic tokens
    counter = 1            # Counter for new prototype names
    unique_chunks = {}
    
    for idx, ch in enumerate(chunks):
        found_match = False
        
        # Check in FORWARD order (oldest first)
        for cname in chunk_order:
            if found_match:
                break
                
            motif = known_chunks[cname]
            
            # Try each variation function
            for vname, func in var_funcs.items():
                try:
                    res = func(motif, ch)
                except Exception:
                    res = None
                    
                if not res:
                    continue
                
                # Normalize result to list of dicts format
                if isinstance(res, dict):
                    res = [res]
                elif isinstance(res, (int, float)):
                    res = [{"position": int(res)}]
                elif isinstance(res, list) and all(isinstance(x, (int, float)) for x in res):
                    res = [{"position": int(x)} for x in res]
                
                # Only care about matches at position 0 (full match)
                for r in res:
                    pos = r.get("position", 0)
                    if pos == 0:
                        # Found first match - use it!
                        token = make_symbol(cname, vname, r)
                        encoding.append(token)
                        
                        # Add this variation to known_chunks UNLESS it's just a repetition
                        if vname != "repetition":
                            known_chunks[token] = ch
                            chunk_order.append(token)
                        
                        found_match = True
                        break
                
                if found_match:
                    break
        
        # No match found - create a new prototype motif
        if not found_match:
            cname = f"c_{counter}"
            known_chunks[cname] = ch
            chunk_order.append(cname)
            encoding.append(f"{cname}")
            counter += 1
            unique_chunks[cname] = ch
    
    return encoding, known_chunks, unique_chunks


# =============================================================================
# NPC CALCULATOR (Latest Version - Separate Pitch/Duration)
# =============================================================================

class NPCCalculator:
    """
    Calculator for Nested Pattern Complexity.
    
    Counts:
    - Chunk complexity: new pitches + new durations + connections (n-1)
    - Variation complexity: first use of function/parameter
    - Sequence complexity: adjacencies
    - Grammar complexity: Sequitur rule adjacencies
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking for a new tune."""
        self.seen_pitches = set()
        self.seen_durations = set()
        self.seen_functions = set()  # G, I, R
        self.seen_parameters = defaultdict(set)  # T: {param1, param2}, A: {2, 3}
        self.seen_change_patterns = set()  # For P, H
        
        self.costs = {
            'chunks': 0,
            'variations': 0,
            'sequences': 0,
            'ho_patterns': 0,
            'au_patterns': 0,
            'grammar': 0,
        }
    
    def calculate_chunk_complexity(self, unique_chunks):
        """
        Calculate complexity for unique chunks.
        
        Chunk cost = new pitches + new durations + (notes - 1)
        
        This captures the intuition that uniform rhythm is simpler.
        """
        total = 0
        
        for name, notes in unique_chunks.items():
            # Skip variation entries
            if not name.startswith('c_'):
                continue
            
            new_pitches = 0
            new_durations = 0
            
            for pitch, duration in notes:
                if pitch not in self.seen_pitches:
                    self.seen_pitches.add(pitch)
                    new_pitches += 1
                
                if duration not in self.seen_durations:
                    self.seen_durations.add(duration)
                    new_durations += 1
            
            connections = len(notes) - 1 if len(notes) > 0 else 0
            chunk_cost = new_pitches + new_durations + connections
            total += chunk_cost
        
        self.costs['chunks'] = total
        return total
    
    def calculate_variation_complexity(self, motifs_dict):
        """
        Calculate complexity for variations.
        
        - G, I, R: first use +1, subsequent 0
        - T, A, D: new parameter +1, seen parameter 0
        - P, H: new change pattern +count, seen pattern 0
        """
        total = 0
        
        for key in motifs_dict.keys():
            # Skip plain chunks
            if key.startswith('c_'):
                continue
            
            # Retrograde or Inversion (non-parametric)
            if key.startswith('G(') or key.startswith('I(') or key.startswith('R('):
                func_type = key[0]
                if func_type not in self.seen_functions:
                    self.seen_functions.add(func_type)
                    total += 1
                continue
            
            # Transposition, Augmentation, Diminution (parametric)
            match = re.match(r'([TAD])\(([^,]+),([^)]+)\)', key)
            if match:
                func_type, motif, param = match.groups()
                param_key = f"{func_type}:{param}"
                if param_key not in self.seen_parameters[func_type]:
                    self.seen_parameters[func_type].add(param_key)
                    total += 1
                continue
            
            # Pitch change or Rhythm change
            match = re.match(r'([PH])\(([^,]+),(.+)\)', key)
            if match:
                func_type, motif, changes = match.groups()
                pattern_key = f"{func_type}:{changes}"
                if pattern_key not in self.seen_change_patterns:
                    self.seen_change_patterns.add(pattern_key)
                    # Count number of changes
                    n_changes = changes.count('note_index')
                    if n_changes == 0:
                        # Simple format: P(c_1,2,67)
                        n_changes = (changes.count(',') // 2) + 1
                    total += max(1, n_changes)
                continue
        
        self.costs['variations'] = total
        return total
    
    def calculate_sequence_complexity(self, encoding):
        """
        Calculate sequence assembly complexity.
        
        Cost = len(encoding) - 1 adjacencies
        """
        cost = len(encoding) - 1 if len(encoding) > 1 else 0
        self.costs['sequences'] = cost
        return cost
    
    def calculate_grammar_complexity(self, encoding):
        """
        Calculate Sequitur grammar complexity.
        
        Each rule costs (rhs_length - 1) for adjacencies.
        """
        if not HAS_SEQUITUR:
            return 0
        
        try:
            grammar = parse(encoding)
            total = 0
            
            for rule in grammar:
                rhs_length = len(grammar[rule])
                if rhs_length > 0:
                    total += rhs_length - 1
            
            self.costs['grammar'] = total
            return total
        except Exception:
            return 0
    
    def calculate_npc(self, encoding, motifs_dict, unique_chunks):
        """
        Calculate complete NPC for a tune.
        
        Returns dict with total NPC and breakdown.
        """
        self.reset()
        
        # Step 1: Chunk complexity
        self.calculate_chunk_complexity(unique_chunks)
        
        # Step 2: Variation complexity
        self.calculate_variation_complexity(motifs_dict)
        
        # Step 3: Sequence complexity
        self.calculate_sequence_complexity(encoding)
        
        # Step 4 & 5: HO and AU (not implemented in current encoder)
        # These stay at 0
        
        # Step 6: Grammar complexity
        self.calculate_grammar_complexity(encoding)
        
        total = sum(self.costs.values())
        
        return {
            'total': total,
            'breakdown': dict(self.costs),
            'unique_pitches': len(self.seen_pitches),
            'unique_durations': len(self.seen_durations),
        }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_npc_analysis(chunks_list, use_degree=True, output_prefix='irish_npc'):
    """
    Run NPC analysis on entire Irish tune dataset.
    
    Args:
        chunks_list: List of tune dicts with 'title' and 'chunks' or 'degree_chunks'
        use_degree: If True, use 'degree_chunks', else use 'chunks'
        output_prefix: Prefix for output CSV files
        
    Returns:
        DataFrame with results
    """
    npc_calc = NPCCalculator()
    results = []
    
    print(f"Processing {len(chunks_list)} tunes...")
    
    for i, tune in enumerate(chunks_list):
        title = tune.get('title', f'tune_{i}')
        
        # Get chunks
        if use_degree:
            chunks = tune.get('degree_chunks', tune.get('chunks', []))
        else:
            chunks = tune.get('chunks', [])
        
        if not chunks:
            continue
        
        try:
            encoding, motifs, unique_chunks = encode_chunks_incremental(chunks, VAR_FUNCS)
            npc_result = npc_calc.calculate_npc(encoding, motifs, unique_chunks)
            
            results.append({
                'title': title,
                'NPC_total': npc_result['total'],
                'NPC_chunks': npc_result['breakdown']['chunks'],
                'NPC_variations': npc_result['breakdown']['variations'],
                'NPC_sequences': npc_result['breakdown']['sequences'],
                'NPC_grammar': npc_result['breakdown']['grammar'],
                'NPC_ho': npc_result['breakdown']['ho_patterns'],
                'NPC_au': npc_result['breakdown']['au_patterns'],
                'unique_pitches': npc_result['unique_pitches'],
                'unique_durations': npc_result['unique_durations'],
                'encoding_length': len(encoding),
                'num_unique_chunks': len(unique_chunks),
                'num_variations': len(motifs) - len(unique_chunks),
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} tunes...")
                
        except Exception as e:
            print(f"  Error processing {title}: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = Path(f"{output_prefix}_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} results to {output_path}")
    
    return df


def print_statistics(df):
    """
    Print detailed statistics for NPC analysis.
    """
    print("\n" + "=" * 70)
    print("NPC STATISTICS FOR IRISH TUNE DATASET")
    print("=" * 70)
    
    # Overall NPC stats
    print("\n--- TOTAL NPC ---")
    print(f"  Count:  {len(df)}")
    print(f"  Mean:   {df['NPC_total'].mean():.2f}")
    print(f"  Std:    {df['NPC_total'].std():.2f}")
    print(f"  Min:    {df['NPC_total'].min()}")
    print(f"  Max:    {df['NPC_total'].max()}")
    print(f"  Median: {df['NPC_total'].median():.2f}")
    
    # Component breakdown
    components = ['NPC_chunks', 'NPC_variations', 'NPC_sequences', 'NPC_grammar', 'NPC_ho', 'NPC_au']
    
    print("\n--- COMPONENT BREAKDOWN ---")
    print(f"{'Component':<18} {'Mean':>10} {'Std':>10} {'Min':>8} {'Max':>8} {'% Total':>10}")
    print("-" * 70)
    
    total_mean = df['NPC_total'].mean()
    for comp in components:
        mean = df[comp].mean()
        std = df[comp].std()
        min_val = df[comp].min()
        max_val = df[comp].max()
        pct = (mean / total_mean * 100) if total_mean > 0 else 0
        
        name = comp.replace('NPC_', '').capitalize()
        print(f"  {name:<16} {mean:>10.2f} {std:>10.2f} {min_val:>8} {max_val:>8} {pct:>8.1f}%")
    
    # Visual distribution
    print("\n--- WHERE DOES COMPLEXITY COME FROM? ---")
    for comp in components:
        mean = df[comp].mean()
        pct = (mean / total_mean * 100) if total_mean > 0 else 0
        bar = '█' * int(pct / 2)
        name = comp.replace('NPC_', '').capitalize()
        print(f"  {name:<12} {bar} {pct:.1f}%")
    
    # Additional stats
    print("\n--- ADDITIONAL STATISTICS ---")
    print(f"  Avg unique pitches:      {df['unique_pitches'].mean():.2f}")
    print(f"  Avg unique durations:    {df['unique_durations'].mean():.2f}")
    print(f"  Avg encoding length:     {df['encoding_length'].mean():.2f}")
    print(f"  Avg unique chunks:       {df['num_unique_chunks'].mean():.2f}")
    print(f"  Avg num variations:      {df['num_variations'].mean():.2f}")
    
    # Top/Bottom tunes
    print("\n--- TOP 5 MOST COMPLEX TUNES ---")
    top5 = df.nlargest(5, 'NPC_total')[['title', 'NPC_total', 'NPC_chunks', 'NPC_variations', 'NPC_grammar']]
    print(top5.to_string(index=False))
    
    print("\n--- TOP 5 LEAST COMPLEX TUNES ---")
    bottom5 = df.nsmallest(5, 'NPC_total')[['title', 'NPC_total', 'NPC_chunks', 'NPC_variations', 'NPC_grammar']]
    print(bottom5.to_string(index=False))
    
    # Correlation analysis
    print("\n--- CORRELATION WITH TOTAL NPC ---")
    for col in ['NPC_chunks', 'NPC_variations', 'NPC_sequences', 'NPC_grammar', 
                'encoding_length', 'num_unique_chunks', 'num_variations']:
        corr = df['NPC_total'].corr(df[col])
        print(f"  {col:<20}: {corr:.3f}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NPC Calculator for Irish Tune Dataset")
    print("=" * 70)
    print("\nUsage in your notebook:")
    print("  %run calculate_irish_npc_complete.py")
    print("  df = run_npc_analysis(chunks_list, use_degree=True)")
    print("  print_statistics(df)")
    print("\nOr for pitch-based analysis:")
    print("  df = run_npc_analysis(chunks_list, use_degree=False)")
