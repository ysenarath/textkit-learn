"""An OpenTelemetry tracing SDK that writes plain files, and readers.

`FileTracerProvider` implements the OpenTelemetry tracing API and writes
each span, as it starts and ends, to JSON-lines files under a directory,
for shared storage without a database, such as NFS used by SLURM jobs on
several machines. `load_spans` and `load_events` read the spans and
events of many runs into DataFrames, and `monitor.RunMonitor` follows
them as they grow, as ``tklearn runs monitor`` does from the command
line. Code instrumented with the
OpenTelemetry API, such as `tklearn.nn.callbacks.OpenTelemetryCallback`,
records into it. It does not import torch.
"""

from tklearn.tracing.load import load_events, load_spans
from tklearn.tracing.sdk import FileTracerProvider, default_run_name

__all__ = [
    "FileTracerProvider",
    "default_run_name",
    "load_events",
    "load_spans",
]
