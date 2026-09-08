class SimpleLinearSchedule:
    """
    Linear learning rate schedule (from initial value to zero),
    simpler than sb3 LinearSchedule.

    :param initial_value: (float or str) The initial value for the schedule
    """

    def __init__(self, initial_value: float | str) -> None:
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial_value

    def __repr__(self) -> str:
        return f"SimpleLinearSchedule(initial_value={self.initial_value})"


def linear_schedule(initial_value: float | str) -> SimpleLinearSchedule:
    """Instantiate a SimpleLinearSchedule (SB3) with a given value.

    Args:
        initial_value: a float or a string containing the starting value.

    Returns:
         A `SimpleLinearSchedule` object.
    """
    return SimpleLinearSchedule(initial_value)