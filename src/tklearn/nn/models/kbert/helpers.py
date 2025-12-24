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
        start = self._final_start + len(self.text) - len(self.text.lstrip())
        end = self._final_end - len(self.text) + len(self.text.rstrip())
        if start >= end:
            start = end
        return Span(start, end)


def inject(
    text: str, injections: list[Injection], spans: list[Span]
) -> tuple[str, list[Span], list[Span]]:
    """Injects text snippets into the original text at specified indices,
    adjusting for collisions with existing spans.

    Collision Logic:
        - A collision occurs if the injection's target index is strictly inside
          an existing span (`span.start < index < span.end`).
        - Resolution is cascading: if a collision occurs, the injection is moved
          to the end of that span.
        - If this new position falls strictly inside a subsequent overlapping span,
          it moves again.
        - Effectively, the injection "slides" to the right until it exits the
          entire chain of overlapping original spans.
        - Injections at exact boundaries (`start` or `end`) are NOT collisions.

    Whitespace Handling:
        - The `injected_spans` returned by this function disregard leading and
          trailing spaces of the injected text.
        - While the full text (including spaces) is inserted into the string,
          the resulting Span object will only cover the non-whitespace content.
        - If an injection is purely whitespace, the span will collapse to a zero-length span
          at the insertion point.

    Args:
        text (str): The original text.
        injections (list[Injection]): List of Injection objects.
        spans (list[Span]): List of existing spans in the original text.

    Returns:
        tuple: (modified_text, updated_original_spans, new_injection_spans)
    """
    # --- Step 1: Resolve Collisions (Logic from previous answer) ---
    # We resolve against original spans.
    # Optimization: Sort spans by start for efficient collision checking.
    sorted_spans = sorted(spans, key=lambda s: s.start)

    for inj in injections:
        curr = inj.target_index

        # Determine strict inclusion in spans
        # Since we are doing this for multiple inputs, we can just scan.
        # For very large datasets, use a binary search or interval tree.
        # Here, a simple loop over sorted spans is reasonably fast.
        for span in sorted_spans:
            if span.start < curr < span.end:
                curr = span.end
                # We don't break immediately; we continue checking
                # in case the new index lands in an overlapping span.

        inj._resolved_index = curr

    # --- Step 2: Sort Injections ---
    # Sort by resolved index.
    # If two injections resolve to the same spot, keep original order (stable sort).
    # We assume 'injections' list order implies priority.
    injections.sort(key=lambda x: x._resolved_index)

    # --- Step 3: Calculate Offsets & Build Text ---

    # We'll build the new text in chunks.
    result_parts = []
    current_text_idx = 0
    cumulative_shift = 0

    # We need a list of "shift events" to help update the original spans later.
    # Event: (at_original_index, amount_to_add)
    shift_events = []

    for inj in injections:
        # 1. Append text from the last injection point up to this one
        # Note: multiple injections might share the same _resolved_index.
        # The slice len will be 0, which is fine.
        segment = text[current_text_idx : inj._resolved_index]
        result_parts.append(segment)

        # 2. Append the injection
        result_parts.append(inj.text)

        # 3. Track where this injection landed for the return value
        # The start is the resolved index + all previous shifts (including segments of original text)
        # Actually, simpler: Current length of result_parts so far
        # But we haven't joined them yet.
        # Math: _resolved_index + cumulative_shift

        final_start = inj._resolved_index + cumulative_shift
        final_end = final_start + len(inj.text)

        inj._final_start = final_start
        inj._final_end = final_end

        # 4. Update counters
        cumulative_shift += len(inj.text)
        current_text_idx = inj._resolved_index

        # Record shift for Step 4
        shift_events.append((inj._resolved_index, len(inj.text)))

    # Append remaining original text
    result_parts.append(text[current_text_idx:])
    final_text = "".join(result_parts)

    # --- Step 4: Update Original Spans ---

    new_spans = []

    # For efficiency, ensure spans are sorted (they are from Step 1)
    # and shift_events are sorted (they are from Step 2)

    for span in sorted_spans:
        shift_amount = 0

        # Calculate how much this span needs to move.
        # Rule: If shift event happens at or before span.start, the span moves.
        for event_idx, amount in shift_events:
            if event_idx <= span.start:
                shift_amount += amount
            else:
                # Since events are sorted by index, we can stop early
                break

        new_spans.append(
            Span(span.start + shift_amount, span.end + shift_amount)
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
