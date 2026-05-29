import asyncio
import json
from datetime import datetime
from typing import Dict, Any


class TelemetryServer:
    """
    Simulates a Telemetry Server using Server-Sent Events (SSE) logic.
    Provides real-time observability into the bandit's decisions.
    """

    def __init__(self):
        self.history = []

    async def stream_event(self, data: Dict[str, Any]):
        """
        Formats and 'pushes' an event.
        In a real web app, this would yield to an HTTP response stream.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "ROUTING_DECISION",
            "data": data,
        }
        self.history.append(event)

        # Professional SSE Formatting
        sse_payload = f"event: {event['event_type']}\ndata: {json.dumps(event)}\n\n"
        print(f"📡 TELEMETRY PUSH:\n{sse_payload}")
        return sse_payload


async def example_telemetry_usage():
    server = TelemetryServer()
    await server.stream_event(
        {
            "arm": 2,
            "strategy": "Aggressive",
            "tokens_saved": 42.5,
            "latency_ms": 120,
            "reward": 1.45,
        }
    )


if __name__ == "__main__":
    asyncio.run(example_telemetry_usage())
