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

"""CLI — text chat, single question, system info."""

import time

import typer
from rich.console import Console
from rich.prompt import Prompt

from app.config import Config
from app.llm import LLM
from app.monitor import format_stats, get_system_stats

console = Console()
app = typer.Typer(name="assistant", help="Jetson Voice Assistant", add_completion=False)


def _load_llm(config: Config) -> LLM:
    llm = LLM(
        model=config.llm.model,
        base_url=config.llm.base_url,
        backend=config.llm.backend,
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
        timeout=config.llm.timeout,
        system_prompt=config.llm.system_prompt,
    )
    if not llm.load():
        console.print("[red]LLM failed to connect[/red]")
        raise typer.Exit(1)
    return llm


def _stream(llm: LLM, text: str, system_prompt: str):
    """Stream LLM response, print tokens. Returns (full_text, tps, ttft, tokens)."""
    ttft = None
    t0 = time.perf_counter()
    full = ""
    meta_final = {}
    for chunk_data in llm.generate_stream(prompt=text, system_prompt=system_prompt):
        content, meta = chunk_data if isinstance(chunk_data, tuple) else (chunk_data, {})
        if meta.get("done"):
            meta_final = meta
        if content:
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            console.print(content, end="", highlight=False)
            full += content
    elapsed = time.perf_counter() - t0
    tokens = meta_final.get("eval_count", 0) or max(1, len(full) // 4)
    tps = tokens / elapsed if elapsed > 0 else 0
    return full, tps, ttft or 0, tokens


@app.command()
def chat(
    model: str | None = typer.Option(None, "--model", "-m"),
    config_path: str | None = typer.Option(None, "--config", "-c"),
):
    """Interactive text chat."""
    config = Config.load(config_path)
    if model:
        config.llm.model = model

    with console.status("[bold]Loading...[/bold]"):
        llm = _load_llm(config)

    label = "llama.cpp" if config.llm.backend == "openai" else "Ollama"
    console.print(f"[green]✓ LLM ({label}: {llm.model})[/green]")
    console.print("\n[dim]'quit' to exit, 'stats' for system info[/dim]\n")

    try:
        while True:
            try:
                text = Prompt.ask("[cyan]You[/cyan]")
                if not text.strip():
                    continue
                if text.strip().lower() == "quit":
                    break
                if text.strip().lower() == "stats":
                    s = get_system_stats()
                    console.print(f"  {format_stats(s)}")
                    continue
                console.print("[magenta]Assistant[/magenta]: ", end="")
                _, tps, ttft, tok = _stream(llm, text, config.llm.system_prompt)
                console.print()
                console.print(
                    f"[dim]  TTFT: {ttft:.0f}ms | {tps:.1f} tok/s | {tok} tokens | {format_stats(get_system_stats())}[/dim]"
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                continue
    finally:
        llm.unload()
        console.print("[yellow]Goodbye![/yellow]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    model: str | None = typer.Option(None, "--model", "-m"),
    config_path: str | None = typer.Option(None, "--config", "-c"),
):
    """Ask a single question."""
    config = Config.load(config_path)
    if model:
        config.llm.model = model
    with console.status("[bold]Loading...[/bold]"):
        llm = _load_llm(config)
    console.print(f"\n[cyan]Q:[/cyan] {question}\n[magenta]A:[/magenta] ", end="")
    _, tps, ttft, tok = _stream(llm, question, config.llm.system_prompt)
    console.print(f"\n\n[dim]TTFT: {ttft:.0f}ms | {tps:.1f} tok/s | {tok} tokens[/dim]")
    llm.unload()


@app.command()
def info():
    """Show system info and dependency status."""
    console.print("\n[bold cyan]System Info[/bold cyan]\n")
    s = get_system_stats()
    console.print(f"  {format_stats(s)}\n")
    config = Config.load()
    try:
        import httpx

        with httpx.Client(timeout=5.0) as c:
            if config.llm.backend == "openai":
                r = c.get(f"{config.llm.base_url.rstrip('/')}/v1/models")
                if r.status_code == 200:
                    models = [m.get("id", "") for m in r.json().get("data", [])]
                    console.print(f"[green]✓ LLM[/green]: {', '.join(models[:3])}")
            else:
                r = c.get(f"{config.llm.base_url.rstrip('/')}/api/tags")
                if r.status_code == 200:
                    names = [m.get("name", "") for m in r.json().get("models", [])]
                    console.print(f"[green]✓ Ollama[/green]: {', '.join(names[:3])}")
    except Exception:
        console.print(f"[red]✗ LLM not running at {config.llm.base_url}[/red]")
    for lib, name in [
        ("faster_whisper", "faster-whisper"),
        ("kokoro_onnx", "kokoro-onnx"),
    ]:
        try:
            __import__(lib)
            console.print(f"[green]✓ {name}[/green]")
        except ImportError:
            console.print(f"[yellow]⚠ {name} not installed[/yellow]")


def main():
    app()


if __name__ == "__main__":
    main()
