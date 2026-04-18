"""
Hierarchical Music Pattern Decoder

A standalone module for decoding/generating music from:
- Sequitur grammar rules
- Anti-Unification (AU) patterns
- Higher-Order (HO) patterns
- Variation functions (T, I, G, R, A, D, P, H)

Pipeline: Grammar → AU expansion → HO expansion → Variation expansion → Melody

"""

import re
from typing import Dict, List, Tuple, Optional, Any

# =============================================================================
# VARIATION FUNCTIONS
# =============================================================================

def transpose(chunk: List[Tuple[int, int]], interval: int) -> List[Tuple[int, int]]:
    """Transpose all pitches by interval (in scale degrees)."""
    return [(p + interval, d) for p, d in chunk]

def invert(chunk: List[Tuple[int, int]], axis: int = 0) -> List[Tuple[int, int]]:
    """Invert pitches around axis."""
    return [(2 * axis - p, d) for p, d in chunk]

def retrograde(chunk: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Reverse the chunk."""
    return list(reversed(chunk))

def augment(chunk: List[Tuple[int, int]], factor: int = 2) -> List[Tuple[int, int]]:
    """Multiply all durations by factor."""
    return [(p, d * factor) for p, d in chunk]

def diminish(chunk: List[Tuple[int, int]], factor: int = 2) -> List[Tuple[int, int]]:
    """Divide all durations by factor."""
    return [(p, d / factor) for p, d in chunk]  # type: ignore

def pitch_change(chunk: List[Tuple[int, int]], changes: Dict[int, int]) -> List[Tuple[int, int]]:
    """
    Change specific pitches at given positions.
    changes: {position: new_pitch}
    """
    result = list(chunk)
    for pos, new_pitch in changes.items():
        if 0 <= pos < len(result):
            result[pos] = (new_pitch, result[pos][1])
    return result

def rhythm_change(chunk: List[Tuple[int, int]], changes: Dict[int, int]) -> List[Tuple[int, int]]:
    """
    Change specific durations at given positions.
    changes: {position: new_duration}
    """
    result = list(chunk)
    for pos, new_dur in changes.items():
        if 0 <= pos < len(result):
            result[pos] = (result[pos][0], new_dur)
    return result


# =============================================================================
# VARIATION FUNCTION REGISTRY
# =============================================================================

VARIATION_FUNCTIONS = {
    'T': transpose,      # T(chunk, interval)
    'I': invert,         # I(chunk, axis)
    'G': retrograde,     # G(chunk)
    'R': retrograde,     # R(chunk) - alias
    'A': augment,        # A(chunk, factor)
    'D': diminish,       # D(chunk, factor)
    'P': pitch_change,   # P(chunk, pos, val, ...)
    'H': rhythm_change,  # H(chunk, pos, val, ...)
}


# =============================================================================
# SYMBOL PARSING
# =============================================================================

def parse_symbol(sym: str) -> dict:
    """
    Parse a symbol into its components.
    
    Examples:
        'c_1' -> {'type': 'chunk', 'name': 'c_1'}
        'T(c_1,5)' -> {'type': 'variation', 'func': 'T', 'inner': 'c_1', 'params': [5]}
        'T(G(c_1),5)' -> {'type': 'variation', 'func': 'T', 'inner': 'G(c_1)', 'params': [5]}
        'ho_1' -> {'type': 'ho', 'name': 'ho_1'}
        'f_1(5)' -> {'type': 'au', 'name': 'f_1', 'args': ['5']}
        'p_1' -> {'type': 'rule', 'name': 'p_1'}
    """
    sym = sym.strip()
    
    # Check for chunk: c_N
    if re.match(r'^c_\d+$', sym):
        return {'type': 'chunk', 'name': sym}
    
    # Check for HO pattern: ho_N
    if re.match(r'^ho_\d+$', sym):
        return {'type': 'ho', 'name': sym}
    
    # Check for AU pattern: f_N(args)
    au_match = re.match(r'^(f_\d+)\((.+)\)$', sym)
    if au_match:
        name, args_str = au_match.groups()
        args = parse_args(args_str)
        return {'type': 'au', 'name': name, 'args': args}
    
    # Check for grammar rule: p_N
    if re.match(r'^p_\d+$', sym):
        return {'type': 'rule', 'name': sym}
    
    # Check for variation function: F(inner, params...)
    var_match = re.match(r'^([A-Z])\((.+)\)$', sym)
    if var_match:
        func, inner_str = var_match.groups()
        # Parse inner and params - need to handle nested parens
        inner, params = parse_variation_args(inner_str)
        return {'type': 'variation', 'func': func, 'inner': inner, 'params': params}
    
    # Unknown - return as-is
    return {'type': 'unknown', 'value': sym}


def parse_args(args_str: str) -> List[str]:
    """Parse comma-separated args, respecting nested parentheses."""
    args = []
    depth = 0
    current = []
    
    for ch in args_str:
        if ch == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            current.append(ch)
    
    if current:
        args.append(''.join(current).strip())
    
    return args


def parse_variation_args(inner_str: str) -> Tuple[str, List]:
    """
    Parse variation function arguments.
    Returns (inner_symbol, [params])
    
    Examples:
        'c_1,5' -> ('c_1', [5])
        'G(c_1),5' -> ('G(c_1)', [5])
        'c_1' -> ('c_1', [])
        'c_1,2,67' -> ('c_1', [2, 67])
    """
    args = parse_args(inner_str)
    
    if len(args) == 0:
        return '', []
    
    inner = args[0]
    params = []
    
    for arg in args[1:]:
        try:
            params.append(int(arg))
        except ValueError:
            try:
                params.append(float(arg))
            except ValueError:
                params.append(arg)
    
    return inner, params


# =============================================================================
# DECODER: EXPAND SEQUITUR
# =============================================================================

def expand_sequitur(grammar: Dict[str, List[str]], start: str = 'p_0') -> List[str]:
    """
    Expand Sequitur grammar rules recursively.
    
    Args:
        grammar: {'p_0': ['p_1', 'p_1'], 'p_1': ['c_1', 'T(c_1,5)']}
        start: Starting rule (default 'p_0')
    
    Returns:
        Flattened list of symbols
    """
    if start not in grammar:
        return [start]
    
    result = []
    for sym in grammar[start]:
        if sym in grammar:
            result.extend(expand_sequitur(grammar, sym))
        else:
            result.append(sym)
    
    return result


# =============================================================================
# DECODER: EXPAND AU PATTERNS
# =============================================================================

def expand_au(sequence: List[str], au_patterns: Dict[str, dict]) -> List[str]:
    """
    Expand AU patterns by substituting variables.
    
    Args:
        sequence: List of symbols (may contain f_k(val1, val2, ...))
        au_patterns: {
            'f_1': {'pattern': ['T(c_1,X1)', 'G(c_1)'], 'variables': ['X1']},
            ...
        }
    
    Returns:
        Expanded sequence with AU patterns replaced
    """
    result = []
    for sym in sequence:
        expanded = expand_au_symbol(sym, au_patterns)
        result.extend(expanded)
    return result


def expand_au_symbol(sym: str, au_patterns: Dict[str, dict]) -> List[str]:
    """Expand a single AU symbol."""
    # Check if it's an AU pattern instance: f_N(args)
    match = re.match(r'^(f_\d+)\((.+)\)$', sym)
    if not match:
        return [sym]
    
    name, args_str = match.groups()
    if name not in au_patterns:
        return [sym]
    
    pattern_info = au_patterns[name]
    pattern = pattern_info['pattern']
    variables = pattern_info['variables']
    
    # Parse arguments
    values = parse_args(args_str)
    
    if len(values) != len(variables):
        return [sym]  # Mismatch, return as-is
    
    # Substitute variables in pattern
    subst = dict(zip(variables, values))
    result = []
    for p in pattern:
        expanded = p
        for var, val in subst.items():
            expanded = expanded.replace(var, val)
        result.append(expanded)
    
    return result


# =============================================================================
# DECODER: EXPAND HO PATTERNS
# =============================================================================

def expand_ho(sequence: List[str], ho_patterns: Dict[str, dict]) -> List[str]:
    """
    Expand HO patterns.
    
    Args:
        sequence: List of symbols (may contain ho_k or T(ho_k, param))
        ho_patterns: {
            'ho_1': {'base_segment': ['c_1', 'c_2', 'G(c_1)']},
            ...
        }
    
    Returns:
        Expanded sequence with HO patterns replaced
    """
    result = []
    for sym in sequence:
        expanded = expand_ho_symbol(sym, ho_patterns)
        result.extend(expanded)
    return result


def expand_ho_symbol(sym: str, ho_patterns: Dict[str, dict]) -> List[str]:
    """Expand a single HO symbol."""
    # Direct HO reference: ho_N
    if sym in ho_patterns:
        return ho_patterns[sym]['base_segment']
    
    # Wrapped HO: T(ho_1,5) or G(ho_1)
    # Must match pattern: FUNC(ho_N) or FUNC(ho_N,param)
    match = re.match(r'^([A-Z]+)\((ho_\d+)(?:,(.+))?\)$', sym)
    if not match:
        return [sym]
    
    wrapper, ho_name, param = match.groups()
    if ho_name not in ho_patterns:
        return [sym]
    
    base = ho_patterns[ho_name]['base_segment']
    if param:
        return [f"{wrapper}({elem},{param})" for elem in base]
    return [f"{wrapper}({elem})" for elem in base]


# =============================================================================
# DECODER: EXPAND VARIATIONS TO CHUNKS
# =============================================================================

def expand_variations(sequence: List[str], chunks: Dict[str, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    Expand variation functions to get actual chunk data.
    
    Args:
        sequence: List of symbols like ['c_1', 'T(c_1,5)', 'G(c_2)']
        chunks: {'c_1': [(60,240), (62,240)], 'c_2': [...]}
    
    Returns:
        List of chunk data (each chunk is a list of (pitch, duration) tuples)
    """
    result = []
    for sym in sequence:
        chunk_data = expand_variation_symbol(sym, chunks)
        result.append(chunk_data)
    return result


def expand_variation_symbol(sym: str, chunks: Dict[str, List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    """Expand a single symbol to chunk data, applying variations recursively."""
    parsed = parse_symbol(sym)
    
    if parsed['type'] == 'chunk':
        name = parsed['name']
        if name in chunks:
            return list(chunks[name])  # Return a copy
        else:
            raise ValueError(f"Unknown chunk: {name}")
    
    elif parsed['type'] == 'variation':
        func = parsed['func']
        inner = parsed['inner']
        params = parsed['params']
        
        # Recursively expand inner
        inner_data = expand_variation_symbol(inner, chunks)
        
        # Apply variation function
        if func == 'T':
            interval = params[0] if params else 0
            return transpose(inner_data, interval)
        elif func == 'I':
            axis = params[0] if params else 0
            return invert(inner_data, axis)
        elif func in ('G', 'R'):
            return retrograde(inner_data)
        elif func == 'A':
            factor = params[0] if params else 2
            return augment(inner_data, factor)
        elif func == 'D':
            factor = params[0] if params else 2
            return diminish(inner_data, factor)
        elif func == 'P':
            # P(chunk, pos1, val1, pos2, val2, ...)
            changes = {}
            for i in range(0, len(params), 2):
                if i + 1 < len(params):
                    changes[params[i]] = params[i + 1]
            return pitch_change(inner_data, changes)
        elif func == 'H':
            # H(chunk, pos1, val1, pos2, val2, ...)
            changes = {}
            for i in range(0, len(params), 2):
                if i + 1 < len(params):
                    changes[params[i]] = params[i + 1]
            return rhythm_change(inner_data, changes)
        else:
            raise ValueError(f"Unknown variation function: {func}")
    
    else:
        raise ValueError(f"Cannot expand to chunk data: {sym}")


# =============================================================================
# DECODER: FLATTEN TO MELODY
# =============================================================================

def flatten_chunks(chunks: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    """Flatten list of chunks into single melody sequence."""
    melody = []
    for chunk in chunks:
        melody.extend(chunk)
    return melody


# =============================================================================
# FULL DECODE PIPELINE
# =============================================================================

def decode(
    grammar: Dict[str, List[str]],
    chunks: Dict[str, List[Tuple[int, int]]],
    ho_patterns: Optional[Dict[str, dict]] = None,
    au_patterns: Optional[Dict[str, dict]] = None,
    start: str = 'p_0'
) -> dict:
    """
    Full decoding pipeline: Grammar → Symbols → Chunks → Melody
    
    Args:
        grammar: Sequitur grammar {'p_0': [...], 'p_1': [...]}
        chunks: Base chunk definitions {'c_1': [(pitch, dur), ...]}
        ho_patterns: HO patterns {'ho_1': {'base_segment': [...]}}
        au_patterns: AU patterns {'f_1': {'pattern': [...], 'variables': [...]}}
        start: Starting grammar rule
    
    Returns:
        dict with intermediate results and final melody
    """
    ho_patterns = ho_patterns or {}
    au_patterns = au_patterns or {}
    
    # Step 1: Expand Sequitur grammar
    after_sequitur = expand_sequitur(grammar, start)
    
    # Step 2: Expand AU patterns
    after_au = expand_au(after_sequitur, au_patterns)
    
    # Step 3: Expand HO patterns
    after_ho = expand_ho(after_au, ho_patterns)
    
    # Step 4: Expand variations to chunk data
    chunk_data = expand_variations(after_ho, chunks)
    
    # Step 5: Flatten to melody
    melody = flatten_chunks(chunk_data)
    
    return {
        'after_sequitur': after_sequitur,
        'after_au': after_au,
        'after_ho': after_ho,
        'chunk_data': chunk_data,
        'melody': melody
    }


def decode_to_melody(
    grammar: Dict[str, List[str]],
    chunks: Dict[str, List[Tuple[int, int]]],
    ho_patterns: Optional[Dict[str, dict]] = None,
    au_patterns: Optional[Dict[str, dict]] = None,
    start: str = 'p_0'
) -> List[Tuple[int, int]]:
    """Convenience function: decode directly to melody."""
    result = decode(grammar, chunks, ho_patterns, au_patterns, start)
    return result['melody']


# =============================================================================
# DECODE WITH REPORT
# =============================================================================

def decode_with_report(
    grammar: Dict[str, List[str]],
    chunks: Dict[str, List[Tuple[int, int]]],
    ho_patterns: Optional[Dict[str, dict]] = None,
    au_patterns: Optional[Dict[str, dict]] = None,
    start: str = 'p_0'
) -> dict:
    """
    Decode with detailed step-by-step report printed.
    
    Returns the same dict as decode().
    """
    ho_patterns = ho_patterns or {}
    au_patterns = au_patterns or {}
    
    print("=" * 70)
    print("DECODING REPORT")
    print("=" * 70)
    
    # Input summary
    print(f"\n{'─' * 70}")
    print("INPUT")
    print(f"{'─' * 70}")
    print(f"Grammar rules: {len(grammar)}")
    for rule, rhs in grammar.items():
        print(f"  {rule} → {' '.join(rhs)}")
    
    print(f"\nChunks defined: {len(chunks)}")
    for name, data in chunks.items():
        print(f"  {name} = {data}")
    
    if ho_patterns:
        print(f"\nHO patterns: {len(ho_patterns)}")
        for name, info in ho_patterns.items():
            print(f"  {name} = {info['base_segment']}")
    
    if au_patterns:
        print(f"\nAU patterns: {len(au_patterns)}")
        for name, info in au_patterns.items():
            print(f"  {name}({', '.join(info['variables'])}) = {info['pattern']}")
    
    # Step 1: Expand Sequitur
    print(f"\n{'─' * 70}")
    print("STEP 1: EXPAND SEQUITUR GRAMMAR")
    print(f"{'─' * 70}")
    after_sequitur = expand_sequitur(grammar, start)
    print(f"Result ({len(after_sequitur)} symbols):")
    print(f"  {after_sequitur}")
    
    # Step 2: Expand AU
    print(f"\n{'─' * 70}")
    print("STEP 2: EXPAND AU PATTERNS")
    print(f"{'─' * 70}")
    after_au = expand_au(after_sequitur, au_patterns)
    if after_au != after_sequitur:
        print(f"Result ({len(after_au)} symbols):")
        print(f"  {after_au}")
    else:
        print("No AU patterns to expand")
    
    # Step 3: Expand HO
    print(f"\n{'─' * 70}")
    print("STEP 3: EXPAND HO PATTERNS")
    print(f"{'─' * 70}")
    after_ho = expand_ho(after_au, ho_patterns)
    if after_ho != after_au:
        print(f"Result ({len(after_ho)} symbols):")
        print(f"  {after_ho}")
    else:
        print("No HO patterns to expand")
    
    # Step 4: Expand variations
    print(f"\n{'─' * 70}")
    print("STEP 4: EXPAND VARIATIONS TO CHUNKS")
    print(f"{'─' * 70}")
    chunk_data = expand_variations(after_ho, chunks)
    print(f"Result ({len(chunk_data)} chunks):")
    for i, ch in enumerate(chunk_data):
        sym = after_ho[i] if i < len(after_ho) else '?'
        print(f"  {sym} → {ch}")
    
    # Step 5: Flatten
    print(f"\n{'─' * 70}")
    print("STEP 5: FLATTEN TO MELODY")
    print(f"{'─' * 70}")
    melody = flatten_chunks(chunk_data)
    print(f"Final melody ({len(melody)} notes):")
    print(f"  {melody}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Grammar rules: {len(grammar)}")
    print(f"After Sequitur: {len(after_sequitur)} symbols")
    print(f"After AU: {len(after_au)} symbols")
    print(f"After HO: {len(after_ho)} symbols")
    print(f"Chunks: {len(chunk_data)}")
    print(f"Notes: {len(melody)}")
    
    return {
        'after_sequitur': after_sequitur,
        'after_au': after_au,
        'after_ho': after_ho,
        'chunk_data': chunk_data,
        'melody': melody
    }


# =============================================================================
# UTILITY: MELODY TO MIDI (Simple)
# =============================================================================

def melody_to_midi_data(
    melody: List[Tuple[int, int]], 
    ticks_per_beat: int = 480,
    velocity: int = 80
) -> List[dict]:
    """
    Convert melody to simple MIDI note data.
    
    Returns list of {'pitch': p, 'start': t, 'duration': d, 'velocity': v}
    """
    notes = []
    current_time = 0
    
    for pitch, duration in melody:
        notes.append({
            'pitch': pitch,
            'start': current_time,
            'duration': duration,
            'velocity': velocity
        })
        current_time += duration
    
    return notes


def save_melody_to_midi(
    melody: List[Tuple[int, int]],
    filename: str,
    ticks_per_beat: int = 480,
    tempo_bpm: int = 120,
    velocity: int = 80
) -> None:
    """
    Save melody to a MIDI file.
    
    Args:
        melody: List of (pitch, duration) tuples
        filename: Output filename (should end with .mid)
        ticks_per_beat: MIDI resolution (default 480)
        tempo_bpm: Tempo in BPM (default 120)
        velocity: Note velocity 0-127 (default 80)
    """
    # MIDI file structure (minimal Type 0)
    # Header chunk
    header = b'MThd'
    header += (6).to_bytes(4, 'big')        # Header length
    header += (0).to_bytes(2, 'big')        # Format type 0
    header += (1).to_bytes(2, 'big')        # Number of tracks
    header += ticks_per_beat.to_bytes(2, 'big')  # Ticks per beat
    
    # Track data
    track_data = bytearray()
    
    # Tempo meta event (at time 0)
    microseconds_per_beat = int(60_000_000 / tempo_bpm)
    track_data += _var_length(0)            # Delta time
    track_data += b'\xff\x51\x03'           # Tempo meta event
    track_data += microseconds_per_beat.to_bytes(3, 'big')
    
    # Note events
    current_time = 0
    events = []
    
    for pitch, duration in melody:
        # Note on
        events.append((current_time, 0x90, pitch, velocity))
        # Note off
        events.append((current_time + duration, 0x80, pitch, 0))
        current_time += duration
    
    # Sort by time
    events.sort(key=lambda x: (x[0], x[1] == 0x90))  # Note offs before note ons at same time
    
    # Write events with delta times
    last_time = 0
    for time, status, pitch, vel in events:
        delta = time - last_time
        track_data += _var_length(delta)
        track_data += bytes([status, pitch, vel])
        last_time = time
    
    # End of track
    track_data += _var_length(0)
    track_data += b'\xff\x2f\x00'
    
    # Track chunk
    track = b'MTrk'
    track += len(track_data).to_bytes(4, 'big')
    track += bytes(track_data)
    
    # Write file
    with open(filename, 'wb') as f:
        f.write(header + track)
    
    print(f"Saved MIDI file: {filename}")


def _var_length(value: int) -> bytes:
    """Encode integer as MIDI variable-length quantity."""
    if value == 0:
        return b'\x00'
    
    result = []
    while value:
        result.append(value & 0x7f)
        value >>= 7
    
    result.reverse()
    for i in range(len(result) - 1):
        result[i] |= 0x80
    
    return bytes(result)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Simple test
    print("Decoder module loaded successfully!")
    print("See decoder_example.py for usage examples.")
