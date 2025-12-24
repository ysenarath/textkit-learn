import random
import string
import timeit

from tklearn.kb.models import Span
from tklearn.nn.models.kbert.helpers import Injection, inject


def generate_data(
    text_len: int,
    num_spans: int,
    num_injections: int,
    collision_probability: float = 0.5,
):
    """Generates random text, spans, and injections."""
    text = "".join(random.choices(string.ascii_letters + " ", k=text_len))

    spans = []
    for _ in range(num_spans):
        start = random.randint(0, text_len - 10)
        end = start + random.randint(1, 10)
        spans.append(Span(start, end))

    injections = []
    for _ in range(num_injections):
        inj_text = f"[{''.join(random.choices(string.ascii_uppercase, k=3))}]"

        if spans and random.random() < collision_probability:
            # Force a collision by picking a point inside a random span
            target_span = random.choice(spans)
            if target_span.end - target_span.start > 1:
                target = random.randint(
                    target_span.start + 1, target_span.end - 1
                )
            else:
                target = target_span.start  # Fallback
        else:
            target = random.randint(0, text_len)

        injections.append(Injection(inj_text, target))

    return text, injections, spans


def run_benchmark():
    print(
        f"{'Scenario':<25} | {'Text Len':<10} | {'Spans':<8} | {'Injections':<10} | {'Avg Time (ms)':<15}"
    )
    print("-" * 80)

    scenarios = [
        ("Small / Low Collision", 1_000, 10, 5, 0.1),
        ("Small / High Collision", 1_000, 20, 10, 0.9),
        ("Medium / Mixed", 50_000, 500, 100, 0.5),
        ("Large / Sparse", 500_000, 1000, 200, 0.1),
        ("Large / Dense Collisions", 500_000, 5000, 1000, 0.9),
    ]

    for name, t_len, n_spans, n_injections, col_prob in scenarios:
        # Prepare data closure
        text, injections, spans = generate_data(
            t_len, n_spans, n_injections, col_prob
        )

        # Define the wrapper for timeit
        def wrapper():
            inject(text, injections, spans)

        # Run 100 loops, take the average
        iterations = 50 if t_len > 100_000 else 1000
        total_time = timeit.timeit(wrapper, number=iterations)
        avg_time_ms = (total_time / iterations) * 1000

        print(
            f"{name:<25} | {t_len:<10} | {n_spans:<8} | {n_injections:<10} | {avg_time_ms:.4f} ms"
        )


if __name__ == "__main__":
    print("Running Benchmarks...\n")
    run_benchmark()
