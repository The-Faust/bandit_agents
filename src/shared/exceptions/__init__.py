from src.shared.exceptions.base_exception import BaseException
from src.shared.exceptions import context_exceptions


def ex(exception: Exception): raise exception
