#!/usr/bin/env python3
"""Compatibility facade and CLI for the modular video-analysis package."""

from video_analysis.models import *
from video_analysis.detection import *
from video_analysis.telemetry import *
from video_analysis.vision import *
from video_analysis.classifier import *
from video_analysis.normalization import *
from video_analysis.live import *
from video_analysis.pipeline import main, process_video


if __name__ == "__main__":
    main()
