"""Inference: the model-facing half of As If.

``worker`` runs as a separate process (inside the inference sandbox from M3
on) and is the only part of this package that touches a socket, and only
llama-server's on loopback. ``client`` is the orchestrator's side of the pipe
and never imports networking code.
"""
