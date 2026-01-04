# Windows AI Agent Toolset v2

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Windows Only](https://img.shields.io/badge/Platform-Windows%2011-lightgrey)](https://www.microsoft.com/windows)

**A minimal, fully-local vision-based AI agent for controlling Windows desktops using tool-calling with a local vision-language model (VLM).**

This is the second iteration of my experimental project to build a reliable "computer use" agent that runs entirely offline on Windows. The focus in v2 has been on solving two major pain points I encountered in v1:

- Accurately rendering the **mouse cursor** (exact shape, hotspot offset, visibility) in every screenshot so the VLM can truly "see" where the pointer is.
- Making everything **DPI-aware and resolution-independent** using a normalized 0–1000 coordinate system.

The result is a single-script prototype that lets a small local VLM (like Qwen2-VL-2B) observe the screen, reason about tasks, and execute low-level mouse/keyboard actions — all without any external Python dependencies.

## Why I'm Building This

I'm fascinated by the emerging "agentic computer use" capabilities in models like Claude, but I want something that:
- Runs **completely locally** (privacy-first, no cloud APIs).
- Works reliably on **real Windows desktops** with high-DPI scaling, multiple monitors in mind (though currently single-monitor only).
- Uses only the **standard library + ctypes** — no `pyautogui`, `mss`, `Pillow`, or other third-party packages.

This project is my playground for iterating toward a robust, open-source alternative that anyone can run on modest hardware.

## Key Features

- **Precise screenshot capture** with overlaid real cursor icon (using `GetCursorInfo`, `DrawIconEx`, hotspot correction).
- **DPI-aware coordinates**: Normalized 0–1000 scale (inclusive) that maps correctly even at 125%+ display scaling.
- **Tool-calling workflow**: OpenAI-compatible chat completions with base64-encoded images and parallel/sequential function calls.
- **Basic action tools**:
  - `move_mouse(x: int, y: int)`
  - `left_click()`
  - `type_text(text: str)`
  - `scroll(direction: str, amount: int)`
- **Scenario-driven testing**: 10 progressive test cases in `test_scenarios.txt`.
- **Zero external dependencies** — pure Python + Windows API via `ctypes`.
- **Debug-friendly**: Optional screenshot dumps and detailed console logging.

## Requirements

- Windows 10/11 (tested on Windows 11 with 125% scaling and Intel iGPU).
- [LM Studio](https://lmstudio.ai/) (or any OpenAI-compatible local server) running on `http://localhost:1234`.
- A vision-language model that supports image inputs and tool calling. I've been using:
  - `qwen/qwen2-vl-2b-instruct` (small, fast, works well on integrated graphics).

No pip installs needed!

## Setup

1. Download and install [LM Studio](https://lmstudio.ai/).
2. Load a compatible VLM (e.g., Qwen2-VL-2B-Instruct).
3. Start the local server (default settings: port 1234, OpenAI API compatibility enabled).
4. Clone this repository:
   ```bash
   git clone https://github.com/wgabrys88/windows-ai-agent-toolset-v2.git
   cd windows-ai-agent-toolset-v2
   ```

## Usage

Run a specific test scenario:

```bash
python script.py test_scenarios.txt <scenario_number>
```

Example:
```bash
python script.py test_scenarios.txt 7
```

This will:
- Load the shared system prompt and the chosen scenario.
- Start the agent loop (up to 60 steps).
- Capture screenshots (with cursor), send to the LLM, execute tool calls, and feed observations back.

Screenshots are optionally dumped to the current directory as `dump_screen_XXXX.png` for debugging (controlled by the `DUMP_SCREENSHOTS` flag in the script).

## Test Scenarios

The file `test_scenarios.txt` contains a shared system prompt plus 10 progressive scenarios that build confidence in basic capabilities:

1. Basic cursor observation
2. Move cursor to screen center
3. Target Notepad++ text area
4. Click at current position
5. Type "hello" in a text editor
6. Move to top-left corner
7. Move to bottom-right corner (verified working at 125% scaling)
8. Scroll test
9. Target window title bar (cursor shape change)
10. Diagonal movement verification

As of January 2026, **scenario 7 has been manually verified as passing** on 125% display scaling with an Intel iGPU.

## How It Works (High-Level)

1. The script sets the process to DPI-aware and queries true desktop resolution.
2. For each agent step:
   - Capture the full desktop to a memory bitmap.
   - Overlay the exact current cursor icon at the correct position/hotspot.
   - Encode the bitmap to PNG in-memory (manual implementation, no dependencies).
   - Send the base64 image + conversation history to the local LLM.
3. Parse tool calls from the response.
4. Execute actions via direct Windows API calls (`SetCursorPos`, `SendInput`, etc.).
5. Append tool observation and repeat.

Everything stays local — no data leaves your machine.

## Current Limitations & Known Issues

This is still an early prototype:
- Single-monitor only. (Even if the system may work with multi-monitor setup I prefer simplicity - development and testing happens on single 1080p monitor.)
- Basic toolset (no right/double-click, drag-and-drop, hotkeys, or window management yet).
- Full-resolution screenshots can be slow on lower-end hardware. (However, even on iGPU - the system runs OK. On NVIDIA GTX 1060 6GB VRAM it is superfast.)
- No automated success detection — success is verified manually or by observation. (This point is a little paradox, for sure, I have to focus on the success definition feature.)
- Cursor overlay doesn't handle animated cursors perfectly. (It may be, I have not tested yet this fully, it is about Paint App when cursor becomes a "pen tool" for example.)

## Future Plans

I'm planning to:
- Add more advanced tools (drag, right-click, keyboard shortcuts).

Contributions and suggestions are very welcome!

## License

MIT License — feel free to use, modify, and share.

## Acknowledgments

Inspired by HuggingFace "smolagents" computer use demos, and various open-source agent projects. Special thanks to the LM Studio team for making local inference so accessible.

---

*Built by [@wgabrys88](https://github.com/wgabrys88) — January 2026*
