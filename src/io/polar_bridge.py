from __future__ import annotations
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
from threading import Thread

from bleak import BleakClient, BleakScanner

# Standard Heart Rate Measurement Characteristic
HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
POLAR_DEFAULT_NAME = "Polar Verity Sense"

def _parse_hr_measurement(data: bytes) -> Optional[int]:
    """
    Parse Bluetooth SIG Heart Rate Measurement value.
    Returns bpm as int, or None if cannot parse / out of plausible range.
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
    return bpm if 20 <= bpm <= 240 else None

@dataclass
class HRWindow:
    # 30 s window: balances responsiveness with stability (WESAD-style)
    seconds: float = 30.0

class PolarVeritySenseHR:
    """
    Persistent-loop wrapper for Bleak on Windows so HR notifications don't target a closed loop.
    Maintains a rolling deque of (timestamp, bpm) limited by a time window (default 30 s).
    """
    def __init__(self, device: Optional[str] = None, avg_window: HRWindow = HRWindow()):
        self.device_query = device or POLAR_DEFAULT_NAME
        self.avg_window = avg_window
        self._deque = deque()  # (timestamp, bpm)

        self._client: Optional[BleakClient] = None
        self._connected: bool = False

        # Dedicated asyncio loop & thread (lazy-started on connect)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None

    # ---------- loop/thread helpers ----------
    def _ensure_loop(self):
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    # ---------- async internals ----------
    async def _a_find_device(self) -> Optional[str]:
        dq = (self.device_query or "").strip()
        if ":" in dq or dq.count("-") >= 5:
            return dq  # looks like an address
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
        # prune outside the window
        cut = now - self.avg_window.seconds
        while self._deque and self._deque[0][0] < cut:
            self._deque.popleft()

    async def _a_connect(self):
        if self._connected:
            return
        address = await self._a_find_device()
        if not address:
            raise RuntimeError(f"Polar device '{self.device_query}' not found. Wake it and retry.")
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

    # ---------- public sync API ----------
    def connect(self):
        self._ensure_loop()
        self._run(self._a_connect())

    def disconnect(self):
        if self._loop is None:
            return
        try:
            self._run(self._a_disconnect())
        except Exception:
            pass

    def close(self):
        """Fully stop the background loop/thread. Call after disconnect()."""
        if self._loop is None:
            return
        try:
            self.disconnect()
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=2.0)
        finally:
            self._thread = None
            self._loop = None

    # ---------- data access ----------
    def sample_count(self) -> int:
        """How many samples are currently inside the rolling window."""
        return len(self._deque)

    def window_span(self) -> float:
        """Seconds covered by the current rolling buffer."""
        if len(self._deque) < 2:
            return 0.0
        return float(self._deque[-1][0] - self._deque[0][0])

    def last_sample_age_sec(self) -> Optional[float]:
        """Age in seconds of the newest HR sample; None if no samples yet."""
        if not self._deque:
            return None
        return float(time.time() - self._deque[-1][0])

    def get_bpm(self, default: Optional[float] = None) -> Optional[float]:
        """
        Returns rolling-avg BPM over the last `avg_window.seconds`,
        or `default` if no samples available.
        """
        if not self._deque:
            return default
        vals = [v for (_, v) in self._deque]
        return float(sum(vals) / max(1, len(vals)))

    def wait_for_samples(
        self,
        timeout: float = 8.0,
        min_count: int = 2,
        default: Optional[float] = None,
        poll_interval: float = 0.2,
    ) -> Tuple[int, float, Optional[float]]:
        """
        Block until at least `min_count` samples arrive (within the rolling window),
        or until `timeout` seconds elapse. Returns (count, elapsed_sec, avg_bpm_or_default).
        """
        start = time.time()
        while True:
            cnt = self.sample_count()
            if cnt >= max(1, min_count):
                break
            if (time.time() - start) >= max(0.0, timeout):
                break
            time.sleep(poll_interval)
        elapsed = time.time() - start
        return self.sample_count(), elapsed, self.get_bpm(default=default)

    def wait_for_window(
        self,
        seconds: float = 30.0,
        timeout: float = 45.0,
        default: Optional[float] = None,
        poll_interval: float = 0.2,
    ) -> Tuple[int, float, Optional[float], float]:
        """
        Block until the rolling buffer covers at least `seconds` span,
        or until `timeout` seconds elapse.
        Returns (count, span_sec, avg_bpm_or_default, elapsed_sec).
        """
        start = time.time()
        target = max(0.0, seconds)
        while True:
            span = self.window_span()
            if span >= target:
                break
            if (time.time() - start) >= max(0.0, timeout):
                break
            time.sleep(poll_interval)
        elapsed = time.time() - start
        return self.sample_count(), self.window_span(), self.get_bpm(default=default), elapsed
