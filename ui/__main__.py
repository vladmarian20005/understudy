"""Mission control: the rig live, and the whole loop from one page.

    python -m ui                       http://127.0.0.1:8000 (opens the browser)
    python -m ui --port 8080 --no-browser
    UI="python -m ui --no-browser" ./understudy.sh --sim       next to robotd, from the launcher

Upload a video of a task and describe it: Muse Spark watches it, SAM 3.1 segments what it touched, the brain does it
with the arms (recorded as an episode) and the judge says whether it worked. Every step, and every thought of Muse's
while it drives the arms, shows up live. The cameras and robotd's state show whether or not a run is going.
Needs robotd (the simulator or the real arms) for the arms, and MODEL_API_KEY in .env for Muse and SAM.
Runs are kept in ui_runs/<time>/.
"""
import argparse
import os
import threading
import webbrowser

from .jobs import HERE, Runner
from .live import Live
from .server import serve


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("UI_PORT", 8000)))
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it from another computer on the network")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--runs", default=os.path.join(HERE, "ui_runs"), help="where runs are kept")
    ap.add_argument("--max-steps", type=int, help="the most plan steps per run (default: brain/plan.py's)")
    ap.add_argument("--no-record", action="store_true", help="don't record runs as episodes (and so don't judge them)")
    a = ap.parse_args()

    from brain import llm
    llm.load_env()  # MODEL_API_KEY for the brain, the judge and (inherited) the SAM pipeline
    live = Live()
    runner = Runner(a.runs, max_steps=a.max_steps, record=not a.no_record)
    httpd = serve(live, runner, a.host, a.port)
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}"
    print(f"[ui] mission control on {url}  (Ctrl+C stops it)", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        live.close()
        httpd.server_close()


if __name__ == "__main__":
    main()
