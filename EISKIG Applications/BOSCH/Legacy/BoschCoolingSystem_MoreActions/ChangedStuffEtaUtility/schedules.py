from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSchedule(ABC):
    """BaseSchedule provides basic functionality for the implementation of new schedules. Each schedule should
    define a value function.
    """

    @abstractmethod
    def value(self, progress_remaining: float) -> float:
        """Calculate the value of the learning rate based on the remaining progess.

        :param progress_remaining: Remaing progress, which is calculcated in the base class: 1 (start), 0 (end).
        :return: Output value.
        """
        raise NotImplementedError("You can only instantiate subclasses of BaseSchedule.")

    def __call__(self, progress_remaining: float) -> float:
        """Take the current progress remaining and return the result of self.value."""
        return self.value(progress_remaining)

    def __repr__(self) -> str:
        """Representation of the Schedule

        :return: String representation.
        """
        return f"{self.__class__.__name__}({', '.join([f'{name}={value}' for name, value in self.__dict__.items()])})"


class LinearSchedule(BaseSchedule):
    """
    Linear interpolation schedule adjusts the learning rate between initial_p and final_p.
    The value is calculated based on the remaining progress, which is between 1 (start) and 0 (end).

    :param initial_p: Initial output value.
    :param final_p: Final output value.
    """

    def __init__(self, initial_p: float, final_p: float):
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, progress_remaining: float) -> float:
        """Calculate the value of the learning rate based on the remaining progess.

        :param progress_remaining: Remaing progress, which is calculcated in the base class: 1 (start), 0 (end).
        :return: Output value.
        """
        return self.final_p + progress_remaining * (self.initial_p - self.final_p)

# custom LR schedulers --> important: add them to .venv\Lib\site-packages\eta_utility\eta_x\common\__init__.py 
# to make them acessible 

class WarmUpSchedule(BaseSchedule):
    """
    Warm up schedule adjusts the learning rate between initial_p and final_p. And the holds final_p
    The value is calculated based on the remaining progress, which is between 1 (start) and 0 (end).
    I is used for continuing the training smoothly 

    :param initial_p: Initial output value.
    :param final_p: Final output value. Reached after progress_remaining is smaller than final_p_reached
    """

    def __init__(self, initial_p: float, final_p: float, final_p_reached = 0.8):
        self.initial_p = initial_p
        self.final_p = final_p
        self.final_p_reached = final_p_reached
        # linear equation calculated apropriatly f(x) = m*x+b
        self.b = (final_p - initial_p*final_p_reached)/(1-final_p_reached)
        self.m = (initial_p-((final_p-initial_p*final_p_reached)/(1-final_p_reached)))

    def value(self, progress_remaining: float) -> float:
        """Calculate the value of the learning rate based on the remaining progess.

        :param progress_remaining: Remaing progress, which is calculcated in the base class: 1 (start), 0 (end).
        :return: Output value.
        """
        
        if progress_remaining<self.final_p_reached:
            return self.final_p
        else:
            return  self.m * progress_remaining + self.b


import math 

class WarmupCosineAnnealingLR(BaseSchedule):
    """Learning rate scheduler with warm-up followed by cosine annealing decay."""

    def __init__(self, begin_lr: float, base_lr: float, final_lr: float, warmup_proportion: float = 0.1):
        """
        :param begin_lr: Initial learning rate.
        :param base_lr: Base learning rate after warm-up.
        :param final_lr: Final learning rate after cosine annealing.
        :param warmup_proportion: Proportion of steps for warm-up (default is 0.1).
        """
        self.begin_lr = begin_lr
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_proportion = warmup_proportion
        self.warmup_steps = 0
        self.total_steps = 0

    def value(self, progress_remaining: float) -> float:
        """Calculate the learning rate based on the remaining progress."""
        if progress_remaining >= 1.0:
            return self.begin_lr
        elif progress_remaining >= 1.0 - self.warmup_proportion:
            return self.begin_lr + (self.base_lr - self.begin_lr) * (1 - progress_remaining) / self.warmup_proportion
        else:
            return self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (1 + math.cos(math.pi * (1 - progress_remaining)))


class WarmupPiecewiseConstantLR(BaseSchedule):
    """Learning rate scheduler with warm-up followed by piecewise constant schedule."""

    def __init__(self, begin_lr: float, base_lr: float, schedule: list, warmup_proportion: float = 0.1):
        """
        :param begin_lr: Initial learning rate.
        :param base_lr: Base learning rate after warm-up.
        :param schedule: List of tuples (progress_remaining, learning_rate) forming a piecewise constant schedule.
        """
        self.begin_lr = begin_lr
        self.base_lr = base_lr
        self.schedule = schedule
        self.warmup_proportion = warmup_proportion
        self.warmup_steps = 0
        self.total_steps = 0

    def value(self, progress_remaining: float) -> float:
        """Calculate the learning rate based on the remaining progress."""
        if progress_remaining >= 1.0:
            return self.begin_lr
        elif progress_remaining >= 1.0 - self.warmup_proportion:
            return self.begin_lr + (self.base_lr - self.begin_lr) * (1 - progress_remaining) / self.warmup_proportion
        else:
            lr = self.base_lr
            scheduled_progress_last = 1
            for scheduled_progress, scheduled_lr in self.schedule:
                if progress_remaining <= scheduled_progress and scheduled_progress<scheduled_progress_last:
                    lr = scheduled_lr
                    scheduled_progress_last = scheduled_progress
            return lr

# Example usage 
# Example usage:
schedule = [
    (0.8, 0.5),
    (0.6, 0.2),
    (0.4, 0.1),
    (0.2, 0.05),
    (0.0, 0.01)
]
scheduler = WarmupPiecewiseConstantLR(begin_lr=0.01, base_lr=0.1, schedule=schedule)
for progress in [1.0, 0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]:
    print(f"Progress Remaining: {progress:.2f}, Learning Rate: {scheduler(progress):.5f}")
print('----------------')

scheduler = WarmupCosineAnnealingLR(begin_lr=0.01, base_lr=0.1, final_lr=0.002)
for progress in [1.0, 0.9, 0.5, 0.1, 0.0]:
    print(f"Progress Remaining: {progress:.2f}, Learning Rate: {scheduler(progress):.5f}")
