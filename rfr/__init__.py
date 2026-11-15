"""rfr - a small deterministic offline renderer with GL and software backends.

The package implements a hierarchical scene graph, Blinn-Phong style vertex
lighting, two interchangeable render backends (ModernGL GPU and a pure NumPy
software rasterizer), a cached render-state machine and a command line entry
point.  Both backends share the exact same lighting math and framebuffer
conventions so their output can be compared pixel by pixel.
"""

from .errors import RenderError, ERROR_CODES

__all__ = ["RenderError", "ERROR_CODES"]
__version__ = "1.0.0"
