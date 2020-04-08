from src.Shared.exceptions import BaseException


class ActionNotInContextException(BaseException):
    def __init__(self, value: any):
        BaseException.__init__(
            message='{} \n    action not in context'.format(self.__class__.__name__),
            value=value
        )
