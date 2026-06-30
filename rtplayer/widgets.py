import threading
import time

import torch
import numpy as np

from .synth import RNNSynth
from .player import AudioEngine

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None


class SynthWidget:
    """
    Tkinter widget for real-time control of the RNN synthesizer.

    Usage in a Jupyter notebook:
        %gui tk
        widget = SynthWidget(
            param_names=["pitch"],
            initial_values=[0.5],
            rnn_model=...,
            dac_model=...,
        )
    """

    def __init__(self, param_names, initial_values,
                 rnn_model, dac_model,
                 chunk_size=16, hop_size=8, right_context=4,
                 frame_samples=512, samplerate=44100,
                 buffer_ms=3000, target_buffer_ms=150,
                 audio_device=None, backend="sounddevice",
                 verbose=False):
        if tk is None:
            raise ImportError("tkinter not available")

        self.param_names = list(param_names)
        self.n_params = len(param_names)
        assert len(initial_values) == self.n_params

        self.synth = RNNSynth(
            rnn_model, dac_model,
            chunk_size=chunk_size, hop_size=hop_size,
            right_context=right_context, frame_samples=frame_samples,
            verbose=verbose)

        self.audio = AudioEngine(samplerate=samplerate,
                                 buffer_maxlen=int(samplerate * buffer_ms / 1000),
                                 backend=backend, verbose=verbose)
        self._audio_device = audio_device
        self.verbose = verbose

        hop_samples = hop_size * frame_samples
        self.target_fill = int(samplerate * target_buffer_ms / 1000)
        self.startup_fill = hop_samples * 3 // 2  # 1.5 hops — enough cushion for rate-limited writer

        self._running = False
        self._thread = None
        # Thread-safe cond cache (written by slider callback on main thread,
        # read by inference thread).  Index i -> float in [0, 1].
        self._current_values = list(initial_values)

        self._build_gui(initial_values)

        # Warm the synth now so Play button has ~94ms less delay
        self.synth.warmup(self._get_cond())

    def _build_gui(self, initial_values):
        self.root = tk.Tk()
        self.root.title("RNN Synthesizer")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.slider_vars = []
        for idx, (name, val) in enumerate(zip(self.param_names, initial_values)):
            frame = ttk.Frame(self.root)
            frame.pack(fill=tk.X, padx=8, pady=2)

            label = ttk.Label(frame, text=name, width=12)
            label.pack(side=tk.LEFT)

            var = tk.DoubleVar(value=val)

            lv = tk.StringVar(value=f"{val:.3f}")

            def on_scale(v_str, _idx=idx, _lv=lv):
                v = float(v_str)
                if self.verbose:
                    print(f"[slider] idx={_idx}  val={v:.4f}  "
                          f"current_values={self._current_values}")
                self._current_values[_idx] = v
                _lv.set(f"{v:.3f}")

            scale = tk.Scale(frame, from_=0.0, to=1.0,
                             resolution=0.001, orient=tk.HORIZONTAL,
                             variable=var, length=400,
                             command=on_scale)
            scale.set(val)   # force slider position
            lv.set(f"{val:.3f}")  # force label text
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

            val_label = ttk.Label(frame, textvariable=lv, width=6)
            val_label.pack(side=tk.LEFT, padx=(4, 0))

            self.slider_vars.append((var, scale))

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=8)

        self.play_btn = ttk.Button(btn_frame, text="Play",
                                   command=self._play)
        self.play_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = ttk.Button(btn_frame, text="Stop",
                                   command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.root, textvariable=self.status_var)
        status_label.pack(pady=(0, 4))

    def _get_cond(self):
        device = next(self.synth.rnn_model.parameters()).device
        return torch.tensor(self._current_values, dtype=torch.float32, device=device)

    def _play(self):
        if self._running:
            return

        # First play: synth was warmed at init (skip warmup).
        # Subsequent plays: reset and re-warm.
        if self.synth.current_pos > 0:
            self.synth.reset()
            self.synth.warmup(self._get_cond())

        cond = self._get_cond()

        # On CUDA: pre-warm compiles decode kernels, then reset for clean state
        if self.synth.device.type == "cuda":
            _ = self.synth.generate_hop(cond)
            self.synth.reset()
            cond = self._get_cond()
            self.synth.warmup(cond)

        # Pre-fill one hop synchronously for immediate audio,
        # then start inference thread and audio simultaneously.
        audio = self.synth.generate_hop(self._get_cond())
        self.audio.buffer.write(audio)

        self._running = True
        self._thread = threading.Thread(target=self._inference_loop,
                                        daemon=True)
        self._thread.start()
        self.audio.start(device=self._audio_device)

        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Playing")

    def _stop(self):
        self._running = False
        self.audio.stop()
        self.synth.reset()

        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")

    def _inference_loop(self):
        _hop = 0
        while self._running:
            if self.audio.buffer.n_filled() < self.target_fill:
                _hop += 1
                cond = self._get_cond()
                if self.verbose and (_hop <= 5 or _hop % 50 == 0):
                    print(f"[infer] hop #{_hop}  cond={cond.tolist()}  "
                          f"n_filled={self.audio.buffer.n_filled()}")
                audio = self.synth.generate_hop(cond)
                self.audio.buffer.write(audio)
            else:
                time.sleep(0.001)

    def _on_close(self):
        self._running = False
        self.audio.stop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def start_gui(self):
        """Convenience: call this to enter the Tkinter event loop."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._on_close()
