class SimpleLinearSchedule:
    """
    Linear learning rate schedule (from initial value to zero),
    simpler than sb3 LinearSchedule.

    :param initial_value: (float or str) The initial value for the schedule
    """

    def __init__(self, initial_value: float | str) -> None:
        # Force conversion to float
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial_value

    def __repr__(self) -> str:
        return f"SimpleLinearSchedule(initial_value={self.initial_value})"


def linear_schedule(initial_value: float | str) -> SimpleLinearSchedule:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: A `SimpleLinearSchedule` object
    """
    return SimpleLinearSchedule(initial_value)