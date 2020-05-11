from src.Shared.exceptions.base_exception import BaseException
from src.Shared.exceptions import context_exceptions


def ex(exception: Exception): raise exception
