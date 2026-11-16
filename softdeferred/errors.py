class RendererError(Exception):
    """A recoverable rendering or output error."""


class SceneConfigError(RendererError):
    """The scene or its referenced assets are invalid."""
