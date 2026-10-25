"""As If: private, offline AI captions for your photo library."""

import os

__version__ = "0.1.0.dev0"

# pydantic runs "plugins" found through package entry points: third-party code
# inside the orchestrator, which is meant to run nothing it didn't ask for.
# Set before anything imports pydantic; an explicit user setting wins.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")
