"""
Telemetry & Resilience Layer for SmartPark Vision.
Publishes MQTT JSON payloads and maintains an offline SQLite event buffer.
"""

import os
import json
import sqlite3
import time
from typing import Dict, Any, List, Optional
import paho.mqtt.client as mqtt

class TelemetryManager:
    def __init__(self, config: Dict[str, Any]):
        self.device_id = config.get("system", {}).get("device_id", "edge_node_north_garage_01")
        self.camera_id = config.get("system", {}).get("camera_id", "cam_zone_A")
        
        tele_cfg = config.get("telemetry", {})
        self.mqtt_enabled = tele_cfg.get("mqtt_enabled", False)
        self.mqtt_broker = tele_cfg.get("mqtt_broker", "localhost")
        self.mqtt_port = tele_cfg.get("mqtt_port", 1883)
        self.occupancy_topic = tele_cfg.get("occupancy_topic", "smartpark/edge/{device_id}/occupancy").format(device_id=self.device_id)
        self.alerts_topic = tele_cfg.get("alerts_topic", "smartpark/edge/{device_id}/alerts").format(device_id=self.device_id)
        
        self.db_path = tele_cfg.get("sqlite_db_path", "data/events.db")
        self._init_sqlite_buffer()

        self.mqtt_client: Optional[mqtt.Client] = None
        self.is_connected = False

        if self.mqtt_enabled:
            self._connect_mqtt()

    def _init_sqlite_buffer(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_buffer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    synced INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def _connect_mqtt(self):
        try:
            self.mqtt_client = mqtt.Client(client_id=f"smartpark_{self.device_id}_{int(time.time())}")
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            self.mqtt_client.loop_start()
            self.is_connected = True
            self._flush_buffered_events()
        except Exception as e:
            self.is_connected = False
            print(f"[Telemetry] MQTT offline: {e}. Buffering telemetry in SQLite.")

    def emit_occupancy(self, spots: List[Dict[str, Any]]):
        """
        Formats and emits the Occupancy Telemetry Output Payload (PRD Section 6.1).
        """
        occupied_count = sum(1 for s in spots if s.get("status") == "occupied")
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device_id": self.device_id,
            "camera_id": self.camera_id,
            "total_spots": len(spots),
            "occupied_spots": occupied_count,
            "spots": spots
        }
        self._dispatch(self.occupancy_topic, payload)

    def emit_violation(self, violation_payload: Dict[str, Any]):
        """
        Formats and emits the Illegal Parking Alert Payload (PRD Section 6.2).
        """
        self._dispatch(self.alerts_topic, violation_payload)

    def _dispatch(self, topic: str, payload_dict: Dict[str, Any]):
        payload_str = json.dumps(payload_dict, indent=2)
        
        # Buffer to SQLite for local audit / offline tolerance
        self._buffer_to_sqlite(topic, payload_str, is_synced=1 if self.is_connected else 0)

        # Publish to MQTT if connected
        if self.is_connected and self.mqtt_client:
            try:
                self.mqtt_client.publish(topic, payload_str, qos=1)
            except Exception as e:
                print(f"[Telemetry] Error publishing to MQTT: {e}")
                self.is_connected = False

    def _buffer_to_sqlite(self, topic: str, payload_str: str, is_synced: int):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO event_buffer (topic, payload, synced)
                    VALUES (?, ?, ?)
                """, (topic, payload_str, is_synced))
                conn.commit()
        except Exception as e:
            print(f"[Telemetry] SQLite write error: {e}")

    def _flush_buffered_events(self):
        if not self.is_connected or not self.mqtt_client:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, topic, payload FROM event_buffer WHERE synced = 0 LIMIT 100")
                rows = cursor.fetchall()
                for row in rows:
                    event_id, topic, payload = row
                    self.mqtt_client.publish(topic, payload, qos=1)
                    cursor.execute("UPDATE event_buffer SET synced = 1 WHERE id = ?", (event_id,))
                conn.commit()
        except Exception as e:
            print(f"[Telemetry] Flush error: {e}")
