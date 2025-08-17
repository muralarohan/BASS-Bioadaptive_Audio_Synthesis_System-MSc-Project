# src/io/polar_bridge.py
from __future__ import annotations
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

# Requires bleak==0.21.1
from bleak import BleakClient, BleakScanner

# Standard Heart Rate Measurement Characteristic
HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
POLAR_DEFAULT_NAME = "Polar Verity Sense"

def _parse_hr_measurement(data: bytes) -> Optional[int]:
    """
    Parse Bluetooth SIG Heart Rate Measurement value.
    Returns bpm as int, or None if cannot parse.
    """
    if not data:
        return None
    flags = data[0]
    hr_16bit = (flags & 0x01) == 0x01
    if hr_16bit:
        if len(data) < 3:
            return None
        bpm = int.from_bytes(data[1:3], byteorder="little")
    else:
        if len(data) < 2:
            return None
        bpm = data[1]
    # Sanity bounds
    if 20 <= bpm <= 240:
        return bpm
    return None

@dataclass
class HRWindow:
    seconds: float = 10.0  # rolling average window

class PolarVeritySenseHR:
    """
    Simple synchronous wrapper around bleak for reading HR notifications.
    Use:
        hr = PolarVeritySenseHR(device="Polar Verity Sense")  # or MAC address
        hr.connect()
        bpm = hr.get_bpm(default=baseline)
        hr.disconnect()
    """
    def __init__(self, device: Optional[str] = None, avg_window: HRWindow = HRWindow()):
        self.device_query = device or POLAR_DEFAULT_NAME
        self.avg_window = avg_window
        self._client: Optional[BleakClient] = None
        self._deque = deque()  # (timestamp, bpm)
        self._connected = False

    # ---------- Async internals ----------
    async def _a_find_device(self) -> Optional[str]:
        """
        Returns address (MAC / BLE address) for the first matching device by name substring or exact address if provided.
        """
        dq = self.device_query.strip()
        # If it looks like an address (has ':'), assume it's an address
        if ":" in dq or dq.count("-") >= 5:
            return dq

        devices = await BleakScanner.discover(timeout=5.0)
        dq_lower = dq.lower()
        for d in devices:
            name = (d.name or "").lower()
            if dq_lower in name or POLAR_DEFAULT_NAME.lower() in name:
                return d.address
        return None

    def _on_hr_notify(self, sender: int, data: bytearray):
        bpm = _parse_hr_measurement(bytes(data))
        if bpm is None:
            return
        now = time.time()
        self._deque.append((now, float(bpm)))
        # prune old
        cut = now - self.avg_window.seconds
        while self._deque and self._deque[0][0] < cut:
            self._deque.popleft()

    async def _a_connect(self):
        if self._connected:
            return
        address = await self._a_find_device()
        if not address:
            raise RuntimeError(f"Polar device '{self.device_query}' not found. Turn it on and try again.")
        self._client = BleakClient(address)
        await self._client.connect()
        await self._client.start_notify(HR_CHAR_UUID, self._on_hr_notify)
        self._connected = True

    async def _a_disconnect(self):
        if self._client and self._connected:
            try:
                await self._client.stop_notify(HR_CHAR_UUID)
            except Exception:
                pass
            try:
                await self._client.disconnect()
            except Exception:
                pass
        self._connected = False
        self._client = None

    # ---------- Public sync API ----------
    def connect(self):
        asyncio.run(self._a_connect())

    def disconnect(self):
        asyncio.run(self._a_disconnect())

    def get_bpm(self, default: Optional[float] = None) -> Optional[float]:
        """
        Returns rolling-avg BPM or default if nothing received yet.
        """
        if not self._deque:
            return default
        vals = [v for (_, v) in self._deque]
        return float(sum(vals) / max(1, len(vals)))
