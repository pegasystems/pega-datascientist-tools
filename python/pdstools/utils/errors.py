class NotApplicableError(ValueError):
    pass

class NotEagerError(ValueError):
    """Operation only possible in eager mode."""

    def __init__(
        self,
        operationType=None,
        defaultmsg="This operation is only possible in eager mode.",
    ):
        if operationType is not None:
            msg = f"{operationType} is only possible in eager mode"
        else:
            msg = defaultmsg
        super().__init__(msg)