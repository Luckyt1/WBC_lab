"""Command-line simulator shutdown, including the headless Isaac Sim hang workaround."""

import os
import sys
import traceback


def run_simulation(main, app, headless):
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if headless:
            # Also used by the local WBC project: Kit teardown can hang after all
            # work has finished. Callers close log writers before returning here.
            os._exit(exit_code)
        app.close()
    if exit_code:
        raise SystemExit(exit_code)
