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
Text Chat — type your message, hear the response.
Keyboard → LLM stream → TTS stream → Speaker

Usage:
  python3 run_text_chat.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import Config
from app.pipeline import stream_and_speak, load_llm, load_tts, print_response_timing
from app.monitor import get_system_stats, format_stats
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()


def main():
    config = Config.load()

    console.print(Panel.fit(
        "[bold cyan]Text Assistant[/bold cyan]\n"
        "Type your message — response is spoken aloud\n"
        "[dim]'quit' to exit · 'stats' for system info · Ctrl-C to quit[/dim]",
        border_style="cyan",
    ))

    console.print("\n[bold]Loading...[/bold]")
    llm = load_llm(config, console)
    if not llm:
        return
    tts = load_tts(config, console)
    console.print("\n[green bold]Ready![/green bold]\n")

    try:
        while True:
            try:
                text = Prompt.ask("[cyan]You[/cyan]")
                if not text.strip():
                    continue
                if text.strip().lower() == "quit":
                    break
                if text.strip().lower() == "stats":
                    console.print(f"  {format_stats(get_system_stats())}")
                    continue

                console.print("[magenta]Assistant[/magenta]: ", end="")
                sys.stdout.flush()

                full_resp, dt_llm, ttft = stream_and_speak(
                    llm, tts, text, config.llm.system_prompt,
                    pa_sink=None,
                    first_chunk_words=config.tts.first_chunk_words,
                    max_chunk_words=config.tts.max_chunk_words,
                )
                console.print()
                print_response_timing(console, full_resp, dt_llm, ttft)

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                continue
    finally:
        llm.unload()
        if tts:
            tts.unload()
        console.print("[yellow]Goodbye![/yellow]")


if __name__ == "__main__":
    main()
