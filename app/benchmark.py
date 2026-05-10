"""Full-pipeline benchmark: TTS → STT → LLM with fixed, reproducible inputs."""

import time

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

# Fixed inputs — never change these so runs stay comparable
BENCH_TTS_TEXT = "The Jetson voice assistant is ready and waiting for your command."
BENCH_LLM_PROMPT = "Reply in exactly one short sentence: what is the capital of France?"


def run_benchmark(cfg, start_server_fn, stop_server_fn, is_running_fn, wait_fn, find_models_fn):
    """Run the benchmark and print results. Returns timings dict."""
    from app.llm import LLM
    from app.stt import STT
    from app.tts import create_tts

    timings = {}

    console.print()

    # ── TTS ──────────────────────────────────────────────────────────
    console.print("[bold]TTS[/bold]  synthesising fixed sentence...")
    console.print(f'  [dim]"{BENCH_TTS_TEXT}"[/dim]')

    tts = create_tts(voice=cfg.tts.voice, speed=cfg.tts.speed, lang=cfg.tts.lang)
    if not tts.load():
        console.print("  [red]✗ TTS failed to load[/red]")
        return None

    t0 = time.perf_counter()
    tts_result = tts.synthesize(BENCH_TTS_TEXT)
    timings["tts"] = time.perf_counter() - t0
    tts.unload()

    if tts_result.get("error"):
        console.print(f"  [red]✗ {tts_result['error']}[/red]")
        return None

    audio_np = tts_result["audio"]
    tts_sr = tts_result["sample_rate"]
    duration_s = len(audio_np) / tts_sr
    console.print(
        f"  [green]✓[/green] {timings['tts']:.2f}s  "
        f"[dim]({duration_s:.1f}s audio @ {tts_sr} Hz)[/dim]"
    )

    # ── STT ──────────────────────────────────────────────────────────
    console.print("\n[bold]STT[/bold]  transcribing TTS output...")
    console.print(f"  [dim]input: {duration_s:.1f}s WAV from TTS above[/dim]")

    stt = STT(
        model=cfg.stt.model,
        device=cfg.stt.device,
        compute_type=cfg.stt.compute_type,
        language=cfg.stt.language,
        beam_size=cfg.stt.beam_size,
    )
    if not stt.load():
        console.print("  [red]✗ STT failed to load[/red]")
        return None

    # Resample to 16 kHz float32 (Whisper's expected format)
    stt_sr = 16000
    audio_f32 = audio_np.astype(np.float32) / 32768.0
    if tts_sr != stt_sr:
        ratio = stt_sr / tts_sr
        new_len = int(len(audio_f32) * ratio)
        indices = np.linspace(0, len(audio_f32) - 1, new_len)
        audio_f32 = np.interp(indices, np.arange(len(audio_f32)), audio_f32)

    t0 = time.perf_counter()
    stt_result = stt.transcribe(audio_f32, sample_rate=stt_sr)
    timings["stt"] = time.perf_counter() - t0
    stt.unload()

    transcript = stt_result.get("text", "").strip()
    match = transcript.lower().strip(".,!?") == BENCH_TTS_TEXT.lower().strip(".,!?")
    match_tag = "[green]exact match[/green]" if match else "[yellow]mismatch[/yellow]"
    console.print(
        f'  [green]✓[/green] {timings["stt"]:.2f}s  {match_tag}\n  [dim]"{transcript}"[/dim]'
    )

    # ── LLM ──────────────────────────────────────────────────────────
    console.print("\n[bold]LLM[/bold]  sending fixed prompt...")
    console.print(f'  [dim]"{BENCH_LLM_PROMPT}"[/dim]')

    server_started = False
    if not is_running_fn():
        models = find_models_fn()
        if not models:
            console.print("  [yellow]⚠ No .gguf models found — skipping LLM benchmark[/yellow]")
        else:
            console.print(f"  Starting llama-server  [dim]{models[0].name}[/dim]...")
            pid = start_server_fn(models[0])
            if pid and wait_fn(timeout=120):
                server_started = True
            else:
                console.print("  [red]✗ llama-server failed to start[/red]")
                stop_server_fn()

    if is_running_fn():
        llm = LLM(
            model=cfg.llm.model,
            base_url=cfg.llm.base_url,
            backend=cfg.llm.backend,
            max_tokens=50,
            temperature=0.0,  # deterministic for reproducibility
        )
        if llm.load():
            ttft = None
            response = []
            t0 = time.perf_counter()
            for chunk, _ in llm.generate_stream(BENCH_LLM_PROMPT):
                if chunk:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    response.append(chunk)
            timings["llm_total"] = time.perf_counter() - t0
            timings["llm_ttft"] = ttft or timings["llm_total"]
            llm.unload()

            console.print(
                f"  [green]✓[/green] TTFT {timings['llm_ttft']:.2f}s  "
                f"total {timings['llm_total']:.2f}s\n"
                f'  [dim]"{"".join(response).strip()}"[/dim]'
            )
        else:
            console.print("  [red]✗ LLM connection failed[/red]")

        if server_started:
            stop_server_fn()

    # ── Summary ──────────────────────────────────────────────────────
    console.print()
    table = Table(box=box.SIMPLE, padding=(0, 2))
    table.add_column("Component", style="bold")
    table.add_column("Time", justify="right")
    table.add_column("Notes", style="dim")

    table.add_row(
        "TTS (synthesis)",
        f"{timings['tts']:.2f}s",
        f"{duration_s:.1f}s audio, {cfg.tts.voice}",
    )
    table.add_row(
        "STT (transcribe)",
        f"{timings['stt']:.2f}s",
        f"whisper {cfg.stt.model}, {cfg.stt.device}",
    )
    if "llm_total" in timings:
        table.add_row("LLM (TTFT)", f"{timings['llm_ttft']:.2f}s", "time to first token")
        table.add_row("LLM (total)", f"{timings['llm_total']:.2f}s", "full response")

    total = timings["tts"] + timings["stt"] + timings.get("llm_total", 0)
    table.add_row("", "", "")
    table.add_row("[cyan]Total[/cyan]", f"[cyan]{total:.2f}s[/cyan]", "TTS + STT + LLM")

    console.print(table)
    return timings
