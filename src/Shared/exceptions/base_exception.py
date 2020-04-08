class BaseException(Exception):
    def __init__(self, message: str, value: any):
        self.data = {
            'message': message,
            'bad_value': value
        }

    def __str__(self):
        return repr(self.data)
