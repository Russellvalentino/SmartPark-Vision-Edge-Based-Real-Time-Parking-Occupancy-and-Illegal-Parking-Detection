"""
Alert Notification & Webhook Dispatcher for SmartPark Vision.
Dispatches real-time violation alerts and evidence snapshots asynchronously
to Discord, Slack, Telegram, or custom webhook endpoints without blocking edge AI inference.
"""

import os
import json
import threading
import time
from typing import Dict, Any, Optional
import urllib.request
import urllib.error

class AlertNotifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("notifications", {})
        self.enabled = self.config.get("webhook_enabled", False)
        self.webhook_url = self.config.get("webhook_url", "")
        self.webhook_type = self.config.get("webhook_type", "discord").lower() # "discord", "slack", "generic"

    def notify(self, violation_payload: Dict[str, Any]):
        """
        Non-blocking asynchronous alert dispatcher.
        """
        if not self.enabled or not self.webhook_url:
            return

        # Run dispatch in a background thread so edge inference loop is not delayed by network I/O
        thread = threading.Thread(
            target=self._send_webhook_sync,
            args=(violation_payload,),
            daemon=True,
            name="WebhookNotifierWorker"
        )
        thread.start()

    def _send_webhook_sync(self, payload: Dict[str, Any]):
        try:
            if self.webhook_type == "discord":
                formatted_body = self._format_discord_embed(payload)
            elif self.webhook_type == "slack":
                formatted_body = self._format_slack_message(payload)
            else:
                formatted_body = payload

            json_data = json.dumps(formatted_body).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=json_data,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'SmartParkVision-EdgeNode/1.0'
                }
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status in [200, 204]:
                    print(f"[Notifier] Dispatched {self.webhook_type.upper()} webhook for event: {payload.get('event_id')}")
        except Exception as e:
            print(f"[Notifier Warning] Failed to dispatch webhook to {self.webhook_url}: {e}")

    def _format_discord_embed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        veh = payload.get("vehicle_details", {})
        return {
            "username": "SmartPark Vision Bot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2554/2554978.png",
            "embeds": [
                {
                    "title": "🚨 ILLEGAL PARKING VIOLATION DETECTED",
                    "description": f"Vehicle has exceeded continuous dwell limit in restricted zone **{payload.get('zone_id')}**.",
                    "color": 15682628, # Red Hex #EF4444
                    "fields": [
                        {"name": "Zone", "value": str(payload.get("zone_id")), "inline": True},
                        {"name": "Vehicle Track ID", "value": f"#{veh.get('track_id')}", "inline": True},
                        {"name": "Vehicle Type", "value": str(veh.get("class", "Car")).capitalize(), "inline": True},
                        {"name": "Dwell Duration", "value": f"**{veh.get('dwell_time_seconds')}s** (Limit: {veh.get('threshold_seconds')}s)", "inline": True},
                        {"name": "Event ID", "value": str(payload.get("event_id")), "inline": True},
                        {"name": "Timestamp", "value": str(payload.get("timestamp")), "inline": True}
                    ],
                    "footer": {
                        "text": "SmartPark Vision Edge Node Telemetry"
                    }
                }
            ]
        }

    def _format_slack_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        veh = payload.get("vehicle_details", {})
        return {
            "text": f"🚨 *ILLEGAL PARKING VIOLATION ALERT* in *{payload.get('zone_id')}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*🚨 ILLEGAL PARKING VIOLATION DETECTED*\n*Zone:* `{payload.get('zone_id')}`\n*Vehicle ID:* `#{veh.get('track_id')}` ({veh.get('class')})\n*Dwell Time:* `{veh.get('dwell_time_seconds')}s` (Threshold: `{veh.get('threshold_seconds')}s`)\n*Timestamp:* `{payload.get('timestamp')}`"
                    }
                }
            ]
        }
