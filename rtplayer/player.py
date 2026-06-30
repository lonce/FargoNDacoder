import subprocess
import threading
import time
from collections import deque

import torch
import numpy as np


class RingBuffer:
    """Thread-safe ring buffer for audio samples."""

    def __init__(self, maxlen=131072):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def write(self, chunk):
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.squeeze().detach().cpu().numpy()
        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        with self.lock:
            self.buf.extend(chunk.tolist())

    def read(self, n):
        with self.lock:
            out = []
            for _ in range(n):
                out.append(self.buf.popleft() if self.buf else 0.0)
        return np.array(out, dtype=np.float32)

    def n_filled(self):
        with self.lock:
            return len(self.buf)

    def clear(self):
        with self.lock:
            self.buf.clear()


class AudioEngine:
    """
    Manages audio output with a RingBuffer.

    Two backends:
      - "sounddevice": uses PortAudio via sounddevice (callback-driven).
      - "aplay": pipes raw PCM to aplay subprocess (ALSA default → PipeWire).

    The callback / writer thread reads from the buffer; silence on underrun.
    """

    def __init__(self, samplerate=44100, channels=1, blocksize=512,
                 buffer_maxlen=262144, backend="sounddevice",
                 verbose=False):
        self.sr = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.backend = backend
        self.verbose = verbose
        self.buffer = RingBuffer(maxlen=buffer_maxlen)
        self.stream = None
        self._proc = None
        self._writer_thread = None
        self._stop_event = threading.Event()

    def start(self, device=None):
        if self.backend == "aplay":
            self._start_aplay()
        else:
            self._start_sounddevice(device)

    def _start_sounddevice(self, device):
        import sounddevice as sd
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.channels,
            callback=self._callback,
            blocksize=self.blocksize,
            dtype=np.float32,
            device=device,
        )
        self.stream.start()
        if self.verbose:
            print(f"[audio] stream started: device={self.stream.device}  "
                  f"sr={self.stream.samplerate}  blocksize={self.stream.blocksize}")

    def _start_aplay(self):
        self._stop_event.clear()
        cmd = [
            "aplay", "-r", str(self.sr), "-c", str(self.channels),
            "-f", "FLOAT_LE", "-t", "raw",
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._writer_thread = threading.Thread(
            target=self._aplay_writer, daemon=True
        )
        self._writer_thread.start()
        if self.verbose:
            print(f"[audio] aplay started: pid={self._proc.pid}  "
                  f"sr={self.sr}  blocksize={self.blocksize}")

    def _aplay_writer(self):
        bs = self.blocksize
        block_time = bs / self.sr  # real-time duration of one block
        while not self._stop_event.is_set():
            samples = self.buffer.read(bs)
            try:
                self._proc.stdin.write(samples.tobytes())
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                break
            time.sleep(block_time)

    def stop(self):
        if self.backend == "aplay":
            self._stop_aplay()
        else:
            self._stop_sounddevice()

    def _stop_sounddevice(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _stop_aplay(self):
        self._stop_event.set()
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2)
            self._writer_thread = None

    def _callback(self, outdata, frames, time_info, status):
        if self.verbose and status:
            print(f"[audio] callback status: {status}")
        samples = self.buffer.read(frames)
        rms = np.sqrt(np.mean(samples**2))
        if self.verbose:
            if not hasattr(self, '_cb_count'):
                self._cb_count = 0
            self._cb_count += 1
            if self._cb_count <= 5:
                print(f"[audio] callback #{self._cb_count}: read {len(samples)} samples "
                      f"rms={rms:.6f}  buffer_left={self.buffer.n_filled()}")
        outdata[:, 0] = samples
