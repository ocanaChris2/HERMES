# hermes/__init__.py
from .model          import HERMES
from .format_sniffer import sniff, sniff_path, NUM_FORMAT_CLASSES

__all__ = ['HERMES', 'sniff', 'sniff_path', 'NUM_FORMAT_CLASSES']
