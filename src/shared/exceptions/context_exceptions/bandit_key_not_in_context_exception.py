from src.shared.exceptions import BaseException


class BanditKeyNotInContextException(BaseException):
    def __init__(self, value: any):
        BaseException.__init__(
            self,
            message='{} \n    Bandit key not in context'.format(self.__class__.__name__),
            value=value
        )
