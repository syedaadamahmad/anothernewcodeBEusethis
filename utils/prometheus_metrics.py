# utils/prometheus_metrics.py
from prometheus_client import Counter, Gauge, Histogram

# ---- Offer Engine Metrics ----
OFFER_REQUESTS_TOTAL = Counter(
    "offer_combiner_total_requests",
    "Total number of Offer Combiner tool invocations"
)

OFFER_FAILURES_TOTAL = Counter(
    "offer_combiner_failures_total",
    "Total failed Offer Combiner tool invocations"
)

OFFER_LATENCY = Histogram(
    "offer_combiner_latency_seconds",
    "Latency of Offer Combiner tool",
    buckets=[0.1, 0.3, 0.5, 1, 2, 5, 10]
)

OFFER_CONCURRENCY = Gauge(
    "offer_combiner_concurrent_requests",
    "Current concurrent Offer Combiner tool executions"
)

# ---- Flight Search Metrics ----
FLIGHT_REQUESTS_TOTAL = Counter(
    "flight_search_total_requests",
    "Total number of flight search requests"
)

FLIGHT_LATENCY = Histogram(
    "flight_search_latency_seconds",
    "Latency of flight search endpoint",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)
