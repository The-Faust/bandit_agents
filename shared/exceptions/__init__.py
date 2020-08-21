from shared.exceptions.context_exceptions import ActionNotInContextException
from shared.exceptions.context_exceptions import BanditKeyNotInContextException


def ex(exception: Exception): raise exception
