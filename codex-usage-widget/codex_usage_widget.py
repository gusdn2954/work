"""Small always-on-top Codex usage meter for Windows."""

import argparse
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime


SETTINGS_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.dirname(os.path.abspath(__file__))),
    "CodexUsageWidget",
    "settings.json",
)


def load_settings():
    defaults = {"opacity": 0.9, "locked_top_left": True, "x": 0, "y": 0}
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as file:
            saved = json.load(file)
        defaults.update({key: saved[key] for key in defaults if key in saved})
    except (OSError, ValueError, TypeError):
        pass
    return defaults


def save_settings(settings):
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)


def find_codex():
    executable = shutil.which("codex")
    if executable:
        return executable
    base = os.environ.get("LOCALAPPDATA", "")
    matches = glob.glob(os.path.join(base, "OpenAI", "Codex", "bin", "*", "codex.exe"))
    if matches:
        return max(matches, key=os.path.getmtime)
    raise RuntimeError("Codex 실행 파일을 찾지 못했습니다.")


class CodexClient:
    def __init__(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [find_codex(), "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            creationflags=flags,
        )
        self.next_id = 0
        self.request("initialize", {
            "clientInfo": {
                "name": "codex_usage_widget",
                "title": "Codex Usage Widget",
                "version": "1.0.0",
            }
        })
        self.send({"method": "initialized"})

    def send(self, message):
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None):
        self.next_id += 1
        request_id = self.next_id
        message = {"method": method, "id": request_id}
        if params is not None:
            message["params"] = params
        self.send(message)
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Codex 연결이 종료되었습니다.")
            reply = json.loads(line)
            if reply.get("id") != request_id:
                continue
            if "error" in reply:
                raise RuntimeError(reply["error"].get("message", "사용량 조회 실패"))
            return reply.get("result", {})

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()


def summarize(result):
    limits = result.get("rateLimitsByLimitId") or {}
    if not limits and result.get("rateLimits"):
        limits = {"codex": result["rateLimits"]}
    windows = {}
    for bucket in limits.values():
        if not isinstance(bucket, dict):
            continue
        for name in ("primary", "secondary"):
            value = bucket.get(name)
            if not isinstance(value, dict) or value.get("usedPercent") is None:
                continue
            minutes = value.get("windowDurationMins")
            if minutes is None:
                continue
            # Prefer the default Codex bucket if two buckets share a duration.
            if minutes not in windows or bucket.get("limitId") == "codex":
                windows[minutes] = value
    return windows


def display_lines(windows):
    lines = []
    for minutes, value in sorted(windows.items()):
        label = "5시간" if minutes == 300 else "주간" if minutes == 10080 else f"{minutes}분"
        remaining = max(0, min(100, 100 - float(value["usedPercent"])))
        reset = value.get("resetsAt")
        reset_text = datetime.fromtimestamp(reset).strftime("%m/%d %H:%M") if reset else ""
        lines.append(f"{label}  {remaining:.0f}% 남음" + (f"   ·   {reset_text} 초기화" if reset_text else ""))
    return lines or ["표시할 사용량 데이터가 없습니다."]


def read_once():
    client = CodexClient()
    try:
        return display_lines(summarize(client.request("account/rateLimits/read")))
    finally:
        client.close()


def run_widget():
    import tkinter as tk

    settings = load_settings()
    root = tk.Tk()
    root.title("Codex 사용량")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", max(0.1, min(1.0, float(settings["opacity"]))))
    root.resizable(False, False)
    root.configure(bg="#181b24")
    x = 0 if settings["locked_top_left"] else int(settings["x"])
    y = 0 if settings["locked_top_left"] else int(settings["y"])
    root.geometry(f"310x151{x:+d}{y:+d}")

    heading = tk.Label(root, text="Codex 사용량", bg="#181b24", fg="#ffffff", font=("Malgun Gothic", 12, "bold"))
    heading.pack(anchor="w", padx=14, pady=(10, 3))
    body = tk.Label(root, text="불러오는 중…", bg="#181b24", fg="#dce3ef", font=("Malgun Gothic", 10), justify="left", anchor="w")
    body.pack(fill="x", padx=14)
    footer = tk.Label(root, text="", bg="#181b24", fg="#8f9aab", font=("Malgun Gothic", 8), anchor="w")
    footer.pack(fill="x", padx=14, pady=(4, 0))

    opacity_row = tk.Frame(root, bg="#181b24")
    opacity_row.pack(fill="x", padx=14, pady=(0, 7))
    opacity_label = tk.Label(opacity_row, text="투명도", bg="#181b24", fg="#b7c2d1", font=("Malgun Gothic", 9))
    opacity_label.pack(side="left")
    opacity_value = tk.Label(opacity_row, text="", bg="#181b24", fg="#dce3ef", font=("Malgun Gothic", 9), width=4)
    opacity_value.pack(side="right")

    slider = tk.Canvas(opacity_row, width=166, height=24, bg="#181b24", highlightthickness=0, bd=0)
    slider.pack(side="right", padx=(4, 7))
    track_left, track_right, track_y = 9, 157, 12
    knob_radius = 7
    track = slider.create_line(track_left, track_y, track_right, track_y, fill="#52627a", width=5)
    active = slider.create_line(track_left, track_y, track_left, track_y, fill="#8ab4ff", width=5)
    knob = slider.create_oval(0, 0, 0, 0, fill="#dce3ef", outline="#ffffff", width=1)
    opacity_dragging = {"active": False}

    def set_opacity(percent, save=False):
        percent = max(10, min(100, int(round(float(percent)))))
        settings["opacity"] = percent / 100
        root.attributes("-alpha", settings["opacity"])
        opacity_value.config(text=f"{percent}%")
        ratio = (percent - 10) / 90
        knob_x = track_left + ratio * (track_right - track_left)
        slider.coords(active, track_left, track_y, knob_x, track_y)
        slider.coords(knob, knob_x - knob_radius, track_y - knob_radius, knob_x + knob_radius, track_y + knob_radius)
        if save:
            save_settings(settings)

    def opacity_start(_event):
        opacity_dragging["active"] = True
        return "break"

    def opacity_drag(_event):
        if not opacity_dragging["active"]:
            return "break"
        ratio = max(0.0, min(1.0, (_event.x - track_left) / (track_right - track_left)))
        set_opacity(10 + ratio * 90)
        return "break"

    def opacity_end(_event):
        if opacity_dragging["active"]:
            opacity_dragging["active"] = False
            save_settings(settings)
        return "break"

    def slider_press(event):
        if knob in slider.find_withtag("current"):
            return opacity_start(event)
        return "break"

    slider.bind("<ButtonPress-1>", slider_press)
    slider.bind("<B1-Motion>", opacity_drag)
    slider.bind("<ButtonRelease-1>", opacity_end)
    slider.tag_bind(knob, "<Enter>", lambda _event: slider.config(cursor="sb_h_double_arrow"))
    slider.tag_bind(knob, "<Leave>", lambda _event: slider.config(cursor="arrow"))
    set_opacity(round(float(settings["opacity"]) * 100))

    events = queue.Queue()
    refresh = threading.Event()
    stop = threading.Event()
    refresh.set()

    def worker():
        client = None
        while not stop.is_set():
            refresh.wait(timeout=10)
            refresh.clear()
            if stop.is_set():
                break
            try:
                if client is None or client.proc.poll() is not None:
                    client = CodexClient()
                lines = display_lines(summarize(client.request("account/rateLimits/read")))
                updated = datetime.now().strftime("%H:%M:%S")
                events.put(("\n".join(lines), f"{updated} 갱신 · 10초 간격 · 우클릭 메뉴"))
            except Exception as exc:
                if client:
                    client.close()
                    client = None
                events.put(("조회 실패", str(exc)))
        if client:
            client.close()

    threading.Thread(target=worker, daemon=True).start()

    def update_ui():
        try:
            while True:
                main, status = events.get_nowait()
                body.config(text=main)
                footer.config(text=status)
        except queue.Empty:
            pass
        if not stop.is_set():
            root.after(200, update_ui)

    menu = tk.Menu(root, tearoff=0)
    menu.add_command(label="새로고침", command=refresh.set)

    locked_choice = tk.BooleanVar(value=bool(settings["locked_top_left"]))

    def toggle_lock():
        settings["locked_top_left"] = locked_choice.get()
        if settings["locked_top_left"]:
            root.geometry("+0+0")
            settings["x"] = 0
            settings["y"] = 0
        save_settings(settings)

    menu.add_checkbutton(label="왼쪽 위 고정", variable=locked_choice, command=toggle_lock)
    menu.add_separator()

    def quit_app():
        stop.set()
        refresh.set()
        root.destroy()

    menu.add_command(label="종료", command=quit_app)
    root.protocol("WM_DELETE_WINDOW", quit_app)

    def show_menu(event):
        menu.tk_popup(event.x_root, event.y_root)
        menu.grab_release()

    move_drag = {}

    def start_move(event):
        if settings["locked_top_left"]:
            return
        move_drag["x"] = event.x_root - root.winfo_x()
        move_drag["y"] = event.y_root - root.winfo_y()

    def move_window(event):
        if not settings["locked_top_left"] and "x" in move_drag:
            next_x = event.x_root - move_drag["x"]
            next_y = event.y_root - move_drag["y"]
            root.geometry(f"{next_x:+d}{next_y:+d}")

    def finish_move(_event):
        if not settings["locked_top_left"] and "x" in move_drag:
            settings["x"] = root.winfo_x()
            settings["y"] = root.winfo_y()
            save_settings(settings)
        move_drag.clear()

    # The slider canvas is intentionally excluded: only its round knob handles left-drag.
    movable_widgets = (root, heading, body, footer, opacity_label, opacity_value)
    for widget in movable_widgets:
        widget.bind("<Button-3>", show_menu)
        widget.bind("<ButtonPress-1>", start_move)
        widget.bind("<B1-Motion>", move_window)
        widget.bind("<ButtonRelease-1>", finish_move)
    slider.bind("<Button-3>", show_menu)

    update_ui()
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Read usage once and exit")
    args = parser.parse_args()
    if args.once:
        print("\n".join(read_once()))
    else:
        run_widget()
