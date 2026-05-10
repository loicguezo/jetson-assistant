#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Voice Chat — speak anytime, dynamic recording.
Mic -> Silero/energy VAD -> STT -> LLM stream -> TTS stream -> Speaker

Usage:
  python3 run_voice_chat.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.audio import find_alsa_device
from app.stt import STT
from app.pipeline import (
    SAMPLE_RATE, MicRecorder, warmup_stt, vad_loop, stream_and_speak, load_silero,
    load_llm, load_tts, print_response_timing,
)
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    config = Config.load()

    console.print(Panel.fit(
        "[bold cyan]Voice Assistant[/bold cyan]\n"
        "Speak anytime — auto-detects speech\n"
        "[dim]Ctrl-C to quit[/dim]",
        border_style="cyan",
    ))

    # ── Audio setup ──────────────────────────────────────────────
    mic_hint = config.audio.input_device or "USB Audio"
    result = find_alsa_device(name_hint=mic_hint)
    if not result:
        # Try any available mic as fallback
        result = find_alsa_device(name_hint="")
    if not result:
        console.print("[red]No mic found! Check 'arecord -l' for available devices.[/red]")
        return
    card, dev, mic_name = result
    hw = f"hw:{card},{dev}"
    console.print(f"  Mic: {hw} ({mic_name})")

    # ── Load models ──────────────────────────────────────────────
    console.print("\n[bold]Loading...[/bold]")

    stt = STT(
        model=config.stt.model, device=config.stt.device,
        compute_type=config.stt.compute_type, language=config.stt.language,
        beam_size=config.stt.beam_size,
    )
    stt.load()
    console.print(f"  ✓ STT (faster-whisper, {config.stt.model})")
    console.print("    CUDA warmup...", end=" ")
    console.print(f"done ({warmup_stt(stt):.1f}s)")

    silero_model = None
    if config.vad.use_silero:
        silero_model = load_silero(console)
    else:
        console.print("  [dim]Silero VAD disabled, using energy-only VAD[/dim]")

    llm = load_llm(config, console)
    if not llm:
        return
    tts = load_tts(config, console)

    # ── Start mic ────────────────────────────────────────────────
    effective_chunk_ms = 32 if silero_model else config.vad.chunk_ms
    mic = MicRecorder(console, chunk_ms=effective_chunk_ms)
    if not mic.start(hw, mic_hint):
        console.print("[red]Cannot start recording! Check mic.[/red]")
        return

    console.print("\n[green bold]Ready — speak anytime![/green bold]\n")

    # ── Main loop ────────────────────────────────────────────────
    try:
        for segment in vad_loop(mic, console, vad_cfg=config.vad, silero=silero_model):
            t_stt = time.perf_counter()
            result = stt.transcribe(segment.audio, sample_rate=SAMPLE_RATE)
            text = result.get("text", "").strip()
            dt_stt = time.perf_counter() - t_stt

            if not text:
                err = result.get("error", "")
                console.print(
                    f"[dim]  (not recognized — {segment.duration:.1f}s, "
                    f"rms={segment.rms:.4f}{', err='+err if err else ''})[/dim]"
                )
                mic.resume()
                continue

            console.print(f'  [green]You:[/green] "{text}"')
            console.print("  [magenta]Assistant:[/magenta] ", end="")
            sys.stdout.flush()

            full_resp, dt_llm, ttft = stream_and_speak(
                llm, tts, text, config.llm.system_prompt, mic.pa_sink,
                first_chunk_words=config.tts.first_chunk_words,
                max_chunk_words=config.tts.max_chunk_words,
            )
            console.print()

            console.print(f"  [dim]STT {dt_stt:.1f}s | ", end="")
            print_response_timing(console, full_resp, dt_llm, ttft, prefix="")

            mic.resume()

    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    finally:
        mic.stop()
        stt.unload()
        llm.unload()
        if tts:
            tts.unload()


if __name__ == "__main__":
    main()
