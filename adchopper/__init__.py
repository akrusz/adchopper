"""adchopper: remove ads from podcast MP3s.

Pipeline: transcribe (timestamps) -> classify ad spans (LLM) -> cut (ffmpeg).
"""

__version__ = "0.1.0"
