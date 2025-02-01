class InvalidActionDataException(Exception):
    def __init__(self, action_data: any) -> None:
        message = f"""
        action data is invalid:
            {action_data}
        """

        super().__init__(message)
