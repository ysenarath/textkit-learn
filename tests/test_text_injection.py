import unittest

from tklearn.kb.models import Span
from tklearn.nn.models.kbert.helpers import Injection, inject


class TestInjection(unittest.TestCase):
    def test_no_injections(self):
        """Test that providing no injections returns original data unchanged."""
        text = "Hello World"
        original_spans = [Span(0, 5)]

        new_text, new_spans, injected_spans = inject(text, [], original_spans)

        self.assertEqual(new_text, text)
        self.assertEqual(new_spans, original_spans)
        self.assertEqual(injected_spans, [])

    def test_clean_insert_start_boundary(self):
        """
        Test injection at exact span start.
        Since 'within' is strictly (start < i < end), i == start is NOT a collision.
        It should PREPEND to the span.
        """
        text = "Hello"
        spans = [Span(0, 5)]
        injections = [Injection(text="X", target_index=0)]

        new_text, new_spans, injected_spans = inject(text, injections, spans)

        # "X" inserted at 0 -> "XHello"
        self.assertEqual(new_text, "XHello")
        # "Hello" shifted right by 1
        self.assertEqual(new_spans, [Span(1, 6)])
        # "X" is at 0
        self.assertEqual(injected_spans, [Span(0, 1)])

    def test_clean_insert_end_boundary(self):
        """Test injection at exact span end (Append)."""
        text = "Hello"
        spans = [Span(0, 5)]
        injections = [Injection(text="X", target_index=5)]

        new_text, new_spans, _ = inject(text, injections, spans)

        # "X" inserted at 5 -> "HelloX"
        self.assertEqual(new_text, "HelloX")
        # Span "Hello" does not shift or expand, it stays (0,5)
        self.assertEqual(new_spans, [Span(0, 5)])

    def test_collision_single_span(self):
        """
        Test simple collision strictly inside a span.
        Logic: 0 < 2 < 5 is True -> Move to end (5).
        """
        text = "Hello"
        spans = [Span(0, 5)]
        injections = [Injection(text="X", target_index=2)]

        new_text, new_spans, _ = inject(text, injections, spans)

        # Should NOT split the span ("HeXllo").
        # Should move to end ("HelloX").
        self.assertEqual(new_text, "HelloX")
        self.assertEqual(new_spans, [Span(0, 5)])

    def test_collision_overlapping_spans(self):
        """
        Test collision with multiple overlapping spans.
        The injection should move to the MAX end of all overlapping spans.

        Text: "ABCDEFG"
        Span 1: "ABCD" (0, 4)
        Span 2: "BCDE" (1, 5)
        Injection at index 2 ('C').

        Analysis:
        - Index 2 is strictly inside Span 1 (0 < 2 < 4).
        - Index 2 is strictly inside Span 2 (1 < 2 < 5).
        - Collision detected with both.
        - Max end is max(4, 5) = 5.
        - Target moves to 5.
        """
        text = "ABCDEFG"
        spans = [Span(0, 4), Span(1, 5)]
        injections = [Injection(text="-", target_index=2)]

        new_text, new_spans, _ = inject(text, injections, spans)

        # Result should be "ABCDE-FG" (inserted at 5)
        self.assertEqual(new_text, "ABCDE-FG")

        # Check Span Shifts:
        # Span 1 (0,4): Unchanged (insertion was after 4? No, insertion moved to 5)
        # Wait: The insertion is now at 5.
        # Span 1 ends at 4. 5 is >= 4. No shift for Span 1. -> (0, 4)
        # Span 2 ends at 5. 5 is >= 5. No shift for Span 2. -> (1, 5)
        self.assertEqual(new_spans, [Span(0, 4), Span(1, 5)])

    def test_collision_overlapping_nested_spans(self):
        """
        Test collision with nested spans where the outer span ends later.

        Text: "Content"
        Span Outer: (0, 7) "Content"
        Span Inner: (2, 5) "nte"
        Injection at 3.

        Analysis:
        - 3 is inside (0, 7).
        - 3 is inside (2, 5).
        - Max end is 7.
        - Move injection to 7.
        """
        text = "Content"
        spans = [Span(0, 7), Span(2, 5)]
        injections = [Injection(text="X", target_index=3)]

        new_text, new_spans, _ = inject(text, injections, spans)

        # "ContentX"
        self.assertEqual(new_text, "ContentX")
        # Neither span shifts because insertion is at the very end
        self.assertEqual(new_spans, [Span(0, 7), Span(2, 5)])

    def test_multi_injection_ordering(self):
        """
        Test that multiple injections are handled correctly,
        shifting subsequent spans properly.
        """
        text = "AB"
        # Span A: (0,1), Span B: (1,2)
        spans = [Span(0, 1), Span(1, 2)]

        # Inject '1' at 0 (Prepend A) -> '1AB'
        # Inject '2' at 2 (Append B)  -> '1AB2'
        injections = [
            Injection(text="1", target_index=0),
            Injection(text="2", target_index=2),
        ]

        new_text, new_spans, _ = inject(text, injections, spans)

        self.assertEqual(new_text, "1AB2")

        # Span A (0,1) shifted by len('1') -> (1, 2)
        # Span B (1,2) shifted by len('1') -> (2, 3)
        # Note: '2' is appended at the end, so it doesn't shift existing spans.
        self.assertEqual(new_spans, [Span(1, 2), Span(2, 3)])


if __name__ == "__main__":
    unittest.main()
