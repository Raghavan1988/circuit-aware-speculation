"""Running recent-acceptance length policy for issue I08."""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

from cas.config import ACTION_LENGTHS


class RecentAcceptancePolicy:
    """Map a rolling draft-token acceptance rate to an allowed action.

    ``bands`` contains inclusive lower bounds.  For example, ``((0, 1),
    (0.5, 4))`` selects 1 below 0.5 and 4 at or above 0.5.  Constructor values
    are frozen for a test run and must be selected using development data only.
    Skip rounds do not update draft-token acceptance statistics.
    """

    def __init__(
        self,
        bands: Sequence[tuple[float, int]],
        *,
        initial_action: int,
        window_size: int = 8,
        action_lengths: Sequence[int] = ACTION_LENGTHS,
    ) -> None:
        allowed = tuple(action_lengths)
        if initial_action not in allowed:
            raise ValueError("initial_action is not allowed")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        normalized = tuple((float(bound), int(action)) for bound, action in bands)
        if not normalized or normalized[0][0] != 0.0:
            raise ValueError("bands must start at an inclusive 0.0 bound")
        if any(not math.isfinite(bound) or not 0 <= bound <= 1 for bound, _ in normalized):
            raise ValueError("band bounds must be finite values in [0, 1]")
        if any(a[0] >= b[0] for a, b in zip(normalized, normalized[1:])):
            raise ValueError("band bounds must be strictly increasing")
        if any(action not in allowed for _, action in normalized):
            raise ValueError("band action is not allowed")
        self.bands = normalized
        self.initial_action = int(initial_action)
        self.window_size = int(window_size)
        self.reset()

    def reset(self) -> None:
        self._history: deque[tuple[int, int]] = deque(maxlen=self.window_size)
        self._last_observed_round: int | None = None

    @property
    def acceptance_rate(self) -> float | None:
        proposed = sum(total for _, total in self._history)
        return sum(accepted for accepted, _ in self._history) / proposed if proposed else None

    def __call__(self, context) -> int:
        if context.round_id == 0:
            self.reset()
        last = context.last_round
        if last is not None and self._last_observed_round != last.round_id:
            realized = int(last.realized_draft_len)
            accepted = int(last.accepted_prefix_len)
            if realized < 0 or not 0 <= accepted <= realized:
                raise ValueError("invalid acceptance counts in last_round")
            if realized > 0:
                self._history.append((accepted, realized))
            self._last_observed_round = int(last.round_id)

        rate = self.acceptance_rate
        if rate is None:
            return self.initial_action
        action = self.bands[0][1]
        for bound, candidate in self.bands:
            if rate < bound:
                break
            action = candidate
        return action
