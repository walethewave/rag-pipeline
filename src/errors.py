"""Our own custom error types.

This is a small example of OOP (Object-Oriented Programming) inheritance. Instead of raising
Python's generic `Exception` everywhere, we define our own `AppError` class, and then three more
specific errors that each INHERIT from it: `NotFoundError(AppError)`, `ValidationError(AppError)`,
`UnsupportedPDFError(AppError)`.

"Inherits from" means "is a more specific version of". A `NotFoundError` IS an `AppError` (and is
also still a normal Python `Exception`) - it just always comes with a fixed code and status_code
already filled in, so the code that raises it doesn't have to repeat those details every time.

Why bother with this instead of just raising `Exception("some message")`? Because main.py has ONE
handler that catches `AppError` (the parent class) and turns it into a JSON error response. Thanks
to inheritance, that single handler automatically catches ALL FOUR of these error types - it
doesn't need to know about `NotFoundError` or `ValidationError` specifically. That's the whole
point of OOP inheritance: write the shared behavior once (on the parent), get it for free on every
child.

Called by: pipeline.py, storage.py, and main.py, whenever something goes wrong that the API caller
needs a clean, specific error message for (not a raw Python traceback).
"""


class AppError(Exception):
    """The parent/base error class. Every other error in this file inherits from this one.

    It bundles three pieces of information that a plain `Exception` doesn't have:
    - code: a short machine-readable label (e.g. "not_found"), useful for frontend code to check
    - message: the human-readable sentence to show the user
    - status_code: which HTTP status to reply with (404, 400, 422, ...)

    main.py's single error handler catches this class and reads these three fields to build the
    JSON response - it never needs to catch the child classes individually.
    """

    def __init__(self, code: str, message: str, status_code: int = 400):
        """Stores the three fields above on `self`, and also calls `super().__init__(message)`.

        `super()` means "the parent class" - here, Python's built-in `Exception`. Calling
        `super().__init__(message)` runs `Exception`'s own setup code, which is what lets this
        error still print and behave like a normal exception (e.g. `str(error)` works) even though
        we've added our own extra fields on top of it.
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class NotFoundError(AppError):
    """Raised when something the caller asked for doesn't exist - e.g. a doc_id or index name that
    was never created. This is a CHILD of AppError: it reuses everything AppError already does, and
    just fixes `code="not_found"` and `status_code=404` so callers don't have to type those every
    time. main.py's error handler turns this into an HTTP 404 response automatically."""

    def __init__(self, message: str):
        """Calls `super().__init__(...)` - that's AppError's own `__init__` above - passing it the
        fixed code/status_code for a "not found" situation, plus whatever message the caller gave."""
        super().__init__(code="not_found", message=message, status_code=404)


class ValidationError(AppError):
    """Raised when the caller's input itself is bad - a disallowed index name, wrong file type, or
    a file that's too big. Another child of AppError, fixing `code="validation_error"` and
    `status_code=400` (HTTP's standard "bad request" status)."""

    def __init__(self, message: str):
        """Same pattern as NotFoundError: hand the fixed code/status_code to AppError's __init__
        via super(), plus the specific message describing what was wrong with the input."""
        super().__init__(code="validation_error", message=message, status_code=400)


class UnsupportedPDFError(AppError):
    """Raised by pipeline.py's load_pdf() when a PDF has no extractable text at all - usually
    because it's a scanned image with no real text layer, which we can't read without OCR (which
    this project doesn't support yet). Fixes `code="unsupported_pdf"` and `status_code=422` (HTTP's
    "the request was well-formed but we can't process it" status)."""

    def __init__(self, message: str = "This PDF has no extractable text. OCR is not supported yet."):
        """Same super() pattern again, with a sensible default message so callers can raise this
        with zero arguments (`raise UnsupportedPDFError()`) and still get a clear explanation."""
        super().__init__(code="unsupported_pdf", message=message, status_code=422)
