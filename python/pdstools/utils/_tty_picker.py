"""Minimal stdlib-only arrow-key TTY picker.

Renders a list of options to stderr and lets the user navigate with the
arrow keys. Returns the selected internal key, or None if the user
cancels (Esc / Ctrl-C). Callers are responsible for checking
``sys.stdin.isatty()`` before invoking — when stdin is not a TTY the
caller should use a plain numeric prompt instead.

No external dependencies. Unix uses ``termios`` + ``tty``; Windows
uses ``msvcrt``.
"""

from __future__ import annotations

import sys

_RESET = "\x1b[0m"
_REVERSE = "\x1b[7m"
_CLEAR_LINE = "\x1b[2K"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"


def _render(out, prompt: str, options: list[tuple[str, str]], idx: int, first: bool) -> None:
    if not first:
        # Move cursor back up to overwrite the previous render
        out.write(f"\x1b[{len(options) + 1}A")
    out.write(f"\r{_CLEAR_LINE}{prompt}\n")
    for i, (_key, label) in enumerate(options):
        out.write(f"\r{_CLEAR_LINE}")
        if i == idx:
            out.write(f"  {_REVERSE}> {label}{_RESET}\n")
        else:
            out.write(f"    {label}\n")
    out.flush()


def _read_key_unix() -> str:
    """Read one logical key. Returns 'up', 'down', 'enter', 'cancel', or ''."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Could be a bare Esc or the start of an escape sequence.
            # Read up to two more bytes non-blockingly.
            import select

            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == "[" and select.select([sys.stdin], [], [], 0.05)[0]:
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "up"
                    if ch3 == "B":
                        return "down"
                    return ""
            return "cancel"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":  # Ctrl-C
            return "cancel"
        if ch in ("k",):
            return "up"
        if ch in ("j",):
            return "down"
        if ch in ("q",):
            return "cancel"
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_windows() -> str:  # pragma: no cover - exercised on Windows only
    import msvcrt

    ch = msvcrt.getch()
    if ch in (b"\xe0", b"\x00"):
        ch2 = msvcrt.getch()
        if ch2 == b"H":
            return "up"
        if ch2 == b"P":
            return "down"
        return ""
    if ch in (b"\r", b"\n"):
        return "enter"
    if ch == b"\x1b":
        return "cancel"
    if ch == b"\x03":
        return "cancel"
    return ""


def pick(prompt: str, options: list[tuple[str, str]]) -> str | None:
    """Show an arrow-key picker.

    Parameters
    ----------
    prompt:
        Text shown above the options list.
    options:
        ``[(internal_key, display_label), ...]``. The selected
        ``internal_key`` is returned.

    Returns
    -------
    The selected internal key, or ``None`` if the user cancelled
    (Esc / Ctrl-C / q). Caller is responsible for checking
    ``sys.stdin.isatty()`` first.
    """
    if not options:
        return None

    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        read_key = _read_key_windows
    else:
        read_key = _read_key_unix

    out = sys.stderr
    idx = 0
    out.write(_HIDE_CURSOR)
    try:
        _render(out, prompt, options, idx, first=True)
        while True:
            key = read_key()
            if key == "up":
                idx = (idx - 1) % len(options)
            elif key == "down":
                idx = (idx + 1) % len(options)
            elif key == "enter":
                return options[idx][0]
            elif key == "cancel":
                return None
            else:
                continue
            _render(out, prompt, options, idx, first=False)
    finally:
        out.write(_SHOW_CURSOR)
        out.flush()
