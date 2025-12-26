import bisect
from dataclasses import dataclass

from tklearn.kb.models import Span


@dataclass
class Injection:
    text: str
    target_index: int

    # Internal fields for processing
    _resolved_index: int = 0
    _final_start: int = 0
    _final_end: int = 0

    @property
    def truncated_span(self) -> Span:
        """Calculates the span of the injected text, excluding leading/trailing whitespace.
        Collapses to a zero-length span if the text is entirely whitespace.
        """
        # Calculate how much whitespace to skip from left and right
        l_len = len(self.text) - len(self.text.lstrip())
        r_len = len(self.text) - len(self.text.rstrip())

        start = self._final_start + l_len
        end = self._final_end - r_len

        # Handle pure whitespace or empty strings (where start > end)
        if start >= end:
            start = end

        return Span(start, end)


def _merge_spans(spans: list[Span]) -> list[Span]:
    """Merges strictly overlapping spans into disjoint intervals.

    Logic aligns with collision rules:
    - Spans that strictly overlap (Start A < End B) are merged.
    - Spans that only touch (Start A == End B) are NOT merged, preserving
      the valid insertion point at the boundary.
    """
    if not spans:
        return []

    # Sort by start, then by end (descending) to grab the widest container first
    sorted_spans = sorted(spans, key=lambda s: (s.start, -s.end))
    merged = []

    current_start, current_end = sorted_spans[0].start, sorted_spans[0].end

    for i in range(1, len(sorted_spans)):
        next_span = sorted_spans[i]

        # Strict overlap check:
        # If next_span starts strictly before current_end, they overlap.
        if next_span.start < current_end:
            current_end = max(current_end, next_span.end)
        else:
            merged.append(Span(current_start, current_end))
            current_start, current_end = next_span.start, next_span.end

    merged.append(Span(current_start, current_end))
    return merged


def inject(
    text: str, injections: list[Injection], spans: list[Span]
) -> tuple[str, list[Span], list[Span]]:
    """Injects text snippets into the original text at specified indices,
    adjusting for collisions with existing spans.

    Complexity: O(N log N + M log M) where N=spans, M=injections.
    """

    if not injections:
        return text, spans, []

    # --- Step 1: Efficient Collision Resolution ---
    # Instead of cascading iteratively, we merge overlaps and check once.

    merged_spans = _merge_spans(spans)
    merged_starts = [s.start for s in merged_spans]

    for inj in injections:
        curr = inj.target_index

        # Find the span that starts before or at the target index.
        # bisect_right returns insertion point to keep list sorted.
        idx = bisect.bisect_right(merged_starts, curr) - 1

        if idx >= 0:
            candidate = merged_spans[idx]
            # Strict inclusion check: start < curr < end
            if candidate.start < curr < candidate.end:
                curr = candidate.end

        inj._resolved_index = curr

    # --- Step 2: Sort Injections ---
    # Stable sort ensures deterministic order for injections at the same index
    injections.sort(key=lambda x: x._resolved_index)

    # --- Step 3: Build Text & Calculate Offsets ---

    result_parts = []
    current_text_idx = 0
    cumulative_shift = 0

    # Store shifts for Step 4: (at_original_index, amount_added)
    injection_shifts = []

    for inj in injections:
        # 1. Append text from the last injection point up to this one
        if current_text_idx < inj._resolved_index:
            result_parts.append(text[current_text_idx : inj._resolved_index])
            current_text_idx = inj._resolved_index

        # 2. Append the injection
        result_parts.append(inj.text)

        inj_len = len(inj.text)

        # 3. Calculate final positions for this injection
        final_start = inj._resolved_index + cumulative_shift
        inj._final_start = final_start
        inj._final_end = final_start + inj_len

        # 4. Track shift
        cumulative_shift += inj_len
        injection_shifts.append((inj._resolved_index, inj_len))

    # Append remaining original text
    result_parts.append(text[current_text_idx:])
    final_text = "".join(result_parts)

    # --- Step 4: Update Original Spans (Linear Sweep) ---

    new_spans = []
    # We must iterate original spans in order to match shifts correctly
    sorted_spans = sorted(spans, key=lambda s: s.start)

    current_shift = 0
    inj_idx = 0
    n_injections = len(injection_shifts)

    for span in sorted_spans:
        # Advance the injection pointer to apply all shifts that occur
        # at or before the start of the current span.
        while inj_idx < n_injections:
            r_index, length = injection_shifts[inj_idx]
            if r_index <= span.start:
                current_shift += length
                inj_idx += 1
            else:
                break

        new_spans.append(
            Span(span.start + current_shift, span.end + current_shift)
        )

    # --- Step 5: Collect Injection Spans ---
    injection_spans = [i.truncated_span for i in injections]

    return final_text, new_spans, injection_spans


def example():
    orig_text = "Hello world context"
    # Spans: "Hello" (0-5), "world" (6-11)
    orig_spans = [Span(0, 5), Span(6, 11)]

    # Inject " brave" at 2 (inside Hello -> should move to 5)
    # Inject " big" at 6 (at start of world -> stays at 6, pushes world)
    # Inject " tiny" at 6 (at start of world -> stays at 6, pushes world)
    to_inject = [
        Injection(" brave", 2),
        Injection(" big", 6),
        Injection(" tiny", 6),
        Injection("apple", 0),
        Injection(" " * 5, 0),
    ]

    final_txt, updated_s, new_s = inject(orig_text, to_inject, orig_spans)

    print(f"Final Text: '{final_txt}'")
    print("\nNew Injection Spans:")
    for s in new_s:
        print(f"  {s} -> '{final_txt[s.start : s.end]}'")

    print("\nUpdated Original Spans:")
    for s in updated_s:
        print(f"  {s} -> '{final_txt[s.start : s.end]}'")


if __name__ == "__main__":
    example()
