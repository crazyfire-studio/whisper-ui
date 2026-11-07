class WhisperUIError(Exception):
    pass


class PipelineError(WhisperUIError):
    """A stage failure.

    If ``user_message`` is given, the job page shows it in place of the
    stage's generic failure label. Pass only text that is safe to display: a
    fixed message, never raw exception or tool output, which can carry
    filesystem paths and remote-controlled text.
    """

    def __init__(self, *args: object, user_message: str | None = None) -> None:
        super().__init__(*args)
        self.user_message = user_message


class PreprocessError(PipelineError):
    pass


class TranscriptionError(PipelineError):
    pass


class AlignmentError(PipelineError):
    pass


class DiarizationError(PipelineError):
    pass


class DownloadError(PipelineError):
    pass


class ExportError(WhisperUIError):
    pass
