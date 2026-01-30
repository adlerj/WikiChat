"""Custom exceptions for streaming."""


class HttpStreamError(Exception):
    """Error during HTTP streaming."""

    pass


class CheckpointError(Exception):
    """Error with checkpoint management."""

    pass


class ParseError(Exception):
    """Error during XML parsing."""

    pass
