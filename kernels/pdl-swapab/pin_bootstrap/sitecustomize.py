"""Activate only explicitly requested isolated pin experiments; fail process closed."""
import os
if os.environ.get('QWEN_INDUCTOR_PIN_MODE'):
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
        import pin_inductor_hook
        pin_inductor_hook.install()
    except BaseException:
        import traceback
        traceback.print_exc()
        # Python otherwise treats a sitecustomize exception as nonfatal.
        os._exit(86)
