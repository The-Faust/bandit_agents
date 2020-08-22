from banditMaker.shared.exceptions.base_exception import BaseException


class ActionNotInContextException(BaseException):
    def __init__(self, value: any):
        BaseException.__init__(
            self,
            message='{} \n    action not in context'.format(self.__class__.__name__),
            value=value
        )
