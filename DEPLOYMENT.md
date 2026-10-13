# Searchless Chess deployment

Public URL: https://deep-chess.ngrok.pro/

Checkout: `/home/yeongnam/searchless-chess-web`, upstream `main` at
`86f1fc54cd4e24d89fc41b601c071e82eb029801`.

## GPU lifecycle

- The web process does not initialize CUDA. The first analysis automatically
  starts an isolated GPU worker for the selected model (default: 270M).
- Exactly one model is resident at a time. Repeated requests reuse that worker.
  Selecting another model replaces it; selection is sent with every request.
  The GPU is shared between visitors, so different model selections can cause
  replacement/reloading, but responses remain tied to the requested model.
- Each analysis and visible user activity renews a 30-minute lease. Activity is
  coalesced to at most once a minute. Status polling and an untouched tab do not
  renew the lease. Returning to the tab counts as activity.
- The reaper checks every five seconds. Expiration or “지금 GPU 해제” terminates
  the worker, releasing model arrays, compiled programs, and the CUDA context.
  A status request/heartbeat cannot start a stopped worker.
- Before loading, `nvidia-smi` checks available VRAM. Conservative estimates are
  1 GiB for 9M variants, 2 GiB for 136M, and 3 GiB for 270M. At least 1 GiB is
  reserved and this app admits at most half the GPU's total capacity. These are
  admission estimates, not a hardware partition; other applications can still
  change free VRAM after the check. Errors stop the worker and are retryable.
- `CHESS_GPU_IDLE_SECONDS` overrides 1800 seconds for tests or local tuning.

## Models

All five official ZIP archives are retained in
`searchless_chess/checkpoints/archives/`; extracted models reside beside them.
The action-value archives also include their EMA parameters. Serving uses the
original `params` checkpoint at step 6400000.

9M, 136M and 270M use the bundled Haiku architecture. The two 9M variants use
Flax-format weights, 31-token embeddings and 78 positions, with a dedicated
Flax predictor preserving their weight structure. Behavioral Cloning predicts
move-selection probabilities, so the UI hides the win-probability bar for it.

Official source: https://github.com/google-deepmind/searchless_chess

## Installation / reproduction

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python download_models.py
```

`requirements-web.txt` lists direct serving dependencies; `requirements-lock.txt`
records the complete verified environment including test/browser tools. CUDA 13
requires a compatible NVIDIA driver; this host uses RTX 5090, driver 580.167.08.
Archives are checked against the official storage MD5 metadata before extraction.

## Services

User service definitions:

- `~/.config/systemd/user/searchless-chess.service`
- `~/.config/systemd/user/searchless-chess-ngrok.service`

The app binds only `127.0.0.1:8888`. ngrok uses the existing credential config and
`deep-chess.ngrok.pro`. Both units are enabled. User lingering is already enabled,
so they start on boot without an interactive login. No server reboot was performed.
The existing Friedman/ngrok service is independent.
The HTTP connection limit is 64 to accommodate simultaneous piece-image
downloads and API calls. GPU inference remains serialized in one model worker.

```sh
systemctl --user status searchless-chess.service searchless-chess-ngrok.service
systemctl --user restart searchless-chess.service
journalctl --user -u searchless-chess.service -f
curl -X POST http://127.0.0.1:8888/api/gpu/unload
```

`/healthz` checks web liveness and checkpoint presence without loading CUDA.
`/api/models` lists models; `/api/gpu` exposes the current lease; analysis and
move POST requests accept `{"fen": "...", "model": "270M"}`.

## Verification

```sh
.venv/bin/python -m pytest tests/test_api.py tests/test_gpu_manager.py -q
.venv/bin/playwright install chromium
.venv/bin/python tests/smoke_test.py https://deep-chess.ngrok.pro
.venv/bin/python tests/gpu_lifecycle.py
.venv/bin/python tests/layout_test.py
```

The smoke test exercises every model's legal-move output, four successive plies,
browser selection/persistence, canvas play, navigation, mobile width, errors,
manual GPU release and automatic reload. Screenshots go to `test-results/`.
Testing requests use the GPU and renew the lease.
The lifecycle test uses a separate worker with a two-second test lease and
checks its PID disappears from `nvidia-smi` after expiry and explicit release;
the production lease remains 1800 seconds.
The layout regression clicks every control at desktop, 500px, 390px and 320px
widths, sampling element bounds throughout loading and error transitions. It
also covers captures, every model, GPU release and retry. Candidate list space,
status text space, hidden color controls and the evaluation bar stay reserved
so asynchronous updates do not collapse the document or jump the scroll position.

Local application changes are not pushed to GitHub. Before pulling updates,
review and preserve the local changes rather than overwriting them.
