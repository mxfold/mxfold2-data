"""Maximum-cardinality subset of reference pairs in EternaFold's grammar.

Only supplied pairs can be retained. Hairpins need >=3 unpaired bases and
single-child loops may contain <=30 unpaired bases; multiloops need >=2 stems.
The interval schedule also removes crossing pairs without adding new pairs.
"""
from bisect import bisect_left

CANONICAL = {'AU', 'UA', 'GC', 'CG', 'GU', 'UG'}


def supported(pairs, min_hairpin=3, max_single=30):
    pairs = sorted(pairs)
    endpoints = [x for p in pairs for x in p]
    if len(set(endpoints)) != len(endpoints):
        return False
    children, stack = {}, []
    for i, j in pairs:
        if i < 1 or j - i - 1 < min_hairpin:
            return False
        while stack and stack[-1][1] < i:
            stack.pop()
        if stack:
            if j >= stack[-1][1]:
                return False
            children.setdefault(stack[-1], []).append((i, j))
        stack.append((i, j))
    for (i, j), inside in children.items():
        if len(inside) == 1:
            p, q = inside[0]
            if p - i - 1 + j - q - 1 > max_single:
                return False
    return True


def best_stems(items, values, require_two=False):
    """Weighted disjoint intervals, with 0/1/>=2 immediate-stem states."""
    items = sorted(items, key=lambda p: (p[1], p[0]))
    n, impossible = len(items), -10**9
    ends = [p[1] for p in items]
    scores = [[0] * (n + 1), [impossible] * (n + 1), [impossible] * (n + 1)]
    trace = [[None] * (n + 1) for _ in range(3)]
    for t, pair in enumerate(items, 1):
        for state in (1, 2):
            scores[state][t] = scores[state][t - 1]
        before = bisect_left(ends, pair[0])
        for state in (0, 1, 2):
            if scores[state][before] == impossible:
                continue
            target = min(2, state + 1)
            value = scores[state][before] + values[pair]
            if value > scores[target][t]:
                scores[target][t] = value
                trace[target][t] = (before, state, pair)
    state = 2 if require_two else max(range(3), key=lambda s: scores[s][n])
    value, selected, t = scores[state][n], [], n
    while t and state:
        step = trace[state][t]
        if step is None:
            t -= 1
        else:
            t, state, pair = step
            selected.append(pair)
    return value, tuple(reversed(selected))


def maximum_subset(pairs, min_hairpin=3, max_single=30):
    pairs = sorted(set(pairs))
    if any(j - i - 1 < min_hairpin for i, j in pairs):
        raise ValueError('Remove individually invalid short pairs before projection')
    if supported(pairs, min_hairpin, max_single):
        return set(pairs)
    values, children = {}, {}
    for pair in sorted(pairs, key=lambda p: (p[1] - p[0], p)):
        i, j = pair
        inside = [p for p in values if i < p[0] < p[1] < j]
        # An arbitrarily long hairpin is valid; feature lengths are capped,
        # whereas C_MAX_SINGLE_LENGTH is a hard grammar constraint.
        best, chosen = 1, ()
        for p, q in sorted(inside):
            if p - i - 1 + j - q - 1 <= max_single and 1 + values[(p, q)] > best:
                best, chosen = 1 + values[(p, q)], ((p, q),)
        multi, stems = best_stems(inside, values, require_two=True)
        if 1 + multi > best:
            best, chosen = 1 + multi, stems
        values[pair], children[pair] = best, chosen
    _, roots = best_stems(pairs, values)
    kept, todo = set(), list(roots)
    while todo:
        pair = todo.pop()
        kept.add(pair)
        todo.extend(children[pair])
    if not kept.issubset(pairs) or not supported(kept, min_hairpin, max_single):
        raise AssertionError('Invalid grammar projection')
    return kept


def prune_record(sequence, partners):
    if len(sequence) != len(partners):
        raise ValueError('Sequence/structure length mismatch')
    original = {(i, int(j)) for i, j in enumerate(partners, 1) if j > i}
    reasons, eligible = {}, set()
    for i, j in sorted(original):
        invalid = []
        if j - i - 1 < 3:
            invalid.append('hairpin_distance_lt_4')
        bases = sequence[i - 1] + sequence[j - 1]
        if set(bases) - set('ACGU'):
            invalid.append('ambiguous_pair_endpoint')
        elif bases not in CANONICAL:
            invalid.append('noncanonical_pair')
        if invalid:
            reasons[(i, j)] = ';'.join(invalid)
        else:
            eligible.add((i, j))
    kept = maximum_subset(eligible)
    for pair in eligible - kept:
        reasons[pair] = 'native_grammar_projection'
    result = [0] * len(sequence)
    for i, j in kept:
        result[i - 1], result[j - 1] = j, i
    return result, reasons
