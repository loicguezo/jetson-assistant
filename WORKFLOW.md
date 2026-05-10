# WORKFLOW.md

How we work on this repo — for AI agents picking up mid-session.

## Git & PR workflow

Branch protection is enabled on `main` — direct pushes are rejected.
Always work on a feature branch and open a PR.

```bash
git checkout -b fix/short-description    # or feat/, test/, docs/
# ... implement, test ...
git add <specific files>
git commit -m "type: short description\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push -u origin <branch>
```

**Create PRs via REST API** — `gh pr create` fails with GraphQL permission errors:

```bash
TOKEN=$(cat ~/.config/gh/hosts.yml | grep 'oauth_token' | head -1 | awk '{print $2}')
curl -s --max-time 10 -X POST \
  -H "Authorization: token $TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/segutwein/jetson-assistant/pulls \
  -d '{"title": "...", "head": "<branch>", "base": "main", "body": "..."}'
```

**Update issues** the same way (PATCH `/issues/<n>`).

## Testing

Always activate the venv first:

```bash
source venv/bin/activate
pytest tests/ -v -s                         # full suite
pytest tests/test_server_lifecycle.py -v    # server stop/start tests
python tests/test_llm_latency.py            # TTFT benchmark (needs running server)
```

Tests require llama-server to be running for integration tests; they auto-skip if not.

**Unit test rule:** For every non-trivial change, write a small unit test before opening the PR.
- Pure logic / sysfs reads → `pytest tests/` (no server needed)
- LLM / TTS / STT behaviour → integration test with `pytest.skip` if server not running
- Run the tests and confirm they pass before `git push`
- Name the file `tests/test_<module>.py`, keep tests focused and fast (< 5s each)

**Research rule:** Before concluding something is impossible or not supported on Jetson,
do a quick web search (NVIDIA forum, Jetson AI Lab docs, PyPI index). Past example: we
concluded "Kokoro TTS can't use CUDA on Jetson" based on ORT error messages — a search
revealed the real cause was two conflicting onnxruntime packages installed simultaneously.
The fix was a 2-line reinstall. When evidence contradicts expectations, search first.

## llama-server lifecycle

```bash
./jetson-assistant stop     # kills tracked PID + all orphaned llama-server processes
./jetson-assistant start    # auto-selects model with 5s countdown, then starts server
./jetson-assistant start --text   # text chat instead of voice
```

**Always restart the server after changing `app/manager.py`** — the server process
is long-lived and caches config in memory. If TTFT looks wrong, check:

```bash
ps aux | grep llama-server | grep -v grep
```

The flags must include `-np 1` and `--reasoning off`. If they don't, the server is stale.

## Known issues & fixes (history)

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| TTFT bimodal (0.4s / 15–28s) | Gemma 4 thinking tokens generated silently | `--reasoning off` in llama-server flags (`app/manager.py`) |
| "Using existing server" after stop | Orphaned process not tracked by PID file | `stop_llama_server()` now scans via `pgrep` and kills all |
| KV-cache misses between turns | 3 parallel slots by default | `-np 1` in llama-server flags |
| LLM no response / `ttft=None` | `max_tokens=128` hit limit → KV-cache corruption | Raised to 512 in `config/settings.yaml` |
| Choppy TTS audio | Kokoro RTF > 1.0 on CPU for small chunks | Fixed by CUDA (RTF 0.14x); `first_chunk_words: 3`, `max_chunk_words: 10` |
| TTS stuck on CPU despite onnxruntime-gpu | Both `onnxruntime` (PyPI) AND `onnxruntime-gpu` (Jetson AI Lab) installed — CPU wheel wins | `pip uninstall -y onnxruntime onnxruntime-gpu` then reinstall only `onnxruntime-gpu` with `--force-reinstall numpy==1.26.4` |
| No audio on BT speaker | `aplay` bypasses PulseAudio | Switched to `paplay` in `app/pipeline.py` |

## Context & memory settings (measured)

| Setting | Value | Notes |
|---------|-------|-------|
| `-c` (ctx) | 8192 | ~100 MB more RAM than 4096; Gemma 4 supports 128k |
| `memory_turns` | 20 | benchmarked: TTFT 0.87s at 20 turns (2.2× baseline), 0.7 GB free |
| `max_tokens` | 512 | response budget; counts toward ctx |
| Realistic tokens/turn | ~100 | 50w user + 30w assistant × 1.3; 20 turns ≈ 2000 tokens |

## Active model

Gemma 4 4B-IT Q4_K_M (`gemma-4-E4B-it-Q4_K_M.gguf`, ~5 GB).
This is a thinking model — `--reasoning off` is essential for voice latency.

## Countdown behavior

All blocking prompts in `./jetson-assistant start` auto-select the default
after 5 seconds. Implemented via `_countdown_wait()` / `confirm_with_countdown()`
in `manage.py`. Required for unattended autostart (issue #3).

## Open issues (as of 2026-05-10)

Priority order we've been following:
1. **#5** Conversation memory — next up
2. **#12** Unit tests for pipeline edge cases
3. **#13** CLI flags for LLM parameters (incl. `--reasoning`)
4. **#21** Live memory/GPU stats per response
5. **#3** Autostart via systemd
