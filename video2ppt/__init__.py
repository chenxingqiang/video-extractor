#!/usr/bin/env python
"""Video Extractor - Extract content from videos and presentations using AI."""

__version__ = '0.3.0'

from .compare import compareImg
from .images2pdf import images2pdf
from .video2ppt import main as video2ppt
from .window2ppt import main as window2ppt
from .smart_window2ppt import main as smart_window2ppt

__all__ = [
    'video2ppt', 
    'window2ppt', 
    'smart_window2ppt',
    'compareImg', 
    'images2pdf', 
    '__version__'
]