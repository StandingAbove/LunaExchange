import os
import time
from contextlib import contextmanager
from datetime import datetime

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
    ConsoleMetricExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


@contextmanager
def timed(metric_name: str):
    span = trace.get_current_span()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        span.set_attribute(metric_name, duration)


def setup_otel(instance_prefix: str):
    """Initialize OpenTelemetry for tracing and metrics."""
    api_key = os.getenv("NEWRELIC_API_KEY")

    instance_id = datetime.utcnow().strftime(f"{instance_prefix}-%Y%m%d")
    resource = Resource(
        attributes={
            "service.name": 'trading-engine',
            "service.environment": os.getenv('environment', 'dev'),
            "service.instance.name": instance_prefix,
            "service.instance.id": instance_id,
        }
    )

    # --- Tracer setup
    if api_key:
        span_exporter = OTLPSpanExporter(
            endpoint="https://otlp.nr-data.net/v1/traces",
            headers={"api-key": api_key},
            compression=Compression.Gzip,
        )
    else:
        # Fallback to console exporter
        print("No NEWRELIC_API_KEY found. Using ConsoleSpanExporter.")
        span_exporter = ConsoleSpanExporter()

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # --- Metrics setup
    if api_key:
        metric_exporter = OTLPMetricExporter(
            endpoint="https://otlp.nr-data.net/v1/metrics",
            headers={"api-key": api_key},
            compression=Compression.Gzip,
            preferred_temporality={},  # default: cumulative
        )
    else:
        print("Using ConsoleMetricExporter for metrics.")
        metric_exporter = ConsoleMetricExporter()

    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))


def flush_otel():
    """Flush and shutdown OpenTelemetry resources."""
    tracer_provider: TracerProvider = trace.get_tracer_provider()
    meter_provider: MeterProvider = metrics.get_meter_provider()

    tracer_provider.force_flush()
    meter_provider.force_flush()

    tracer_provider.shutdown()
    meter_provider.shutdown()
