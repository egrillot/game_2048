import time
import numpy as np

from typing import Dict


class ProgressBar:
    """Display a progress bar."""

    def __init__(
        self,
        target: int=None,
        width: int=30,
        interval: float=0.05
    ) -> None:
        """Init the progress bar.
        
        Inputs:
        target: total number of steps expected, None if unknown (int)
        width: progress bar width on screen, defaut value: 30 (int)
        interval: Minimum visual progress update interval in seconds, default value: 0.05 (float).
        """
        self.target = target
        self.width = width
        self.interval = interval
        self._start = time.time()
        self._last_update = 0
        self._time_at_epoch_start = self._start
        self._time_at_epoch_end = None
        self._time_after_first_step = None
        self._total_width = 0

    def update(self, values: Dict[str, float], step: int, finalize: bool=None) -> None:
        """Update the progress bar.
        
        Inputs:
        values: tracked current variables values (Dict[str, float])
        step: current step index (int)
        finalize: whether this is the last update for the progress bar or not. If finalize=None, default value is to current >= self.target (bool).
        """
        if finalize is None:

            if self.target is None:

                finalize = False

            else:

                finalize = step >= self.target

        now = time.time()
        message = ''
        info = ' - %.0fs' % (now - self._start)

        if step == self.target:

            self._time_at_epoch_end = now
        
        if now - self._last_update < self.interval and not finalize:

            return
        
        prev_total_width = self._total_width
        message += '\b' * prev_total_width
        message += '\r'

        if self.target is not None:

            numdigits = int(np.log10(self.target)) + 1
            bar = ('%' + str(numdigits) + 'd/%d [') % (step, self.target)
            prog = float(step) / self.target
            prog_width = int(self.width * prog)

            if prog_width > 0:

                bar += ('=' * (prog_width - 1))

            if step < self.target:

                bar += '>'

            else:

                bar += '='

            bar += ('.' * (self.width - prog_width))
            bar += ']'

        else:

            bar = '%7d/Unknown' % step

        self._total_width = len(bar)
        message += bar
        time_per_unit = self._estimate_step_duration(step, now)

        if self.target is None or finalize:

            info += self._format_time(time_per_unit, 'iteration')

        else:
    
            eta = time_per_unit * (self.target - step)

            if eta > 3600:

                eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)

            elif eta > 60:

                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:

                eta_format = '%ds' % eta

            info = ' - estimated time remaining: %s' % eta_format
        
        for name, value in values.items():

            info += ' - %s:' % name + ' %s' % value
        
        info += '     '

        self._total_width += len(info)

        if prev_total_width > self._total_width:

            info += (' ' * (prev_total_width - self._total_width))

        if finalize:

            info += '\n'

        message += info
        self._last_update = now
        
        print(message, end='\r')

    def _format_time(self, time_per_unit: float, unit_name: str) -> str:
        """Format a given duration to display to the user.
        Output:
        a string with the correctly formatted duration and units
        """
        formatted = ''

        if time_per_unit >= 1 or time_per_unit == 0:

            formatted += ' %.0fs/%s' % (time_per_unit, unit_name)

        elif time_per_unit >= 1e-3:

            formatted += ' %.0fms/%s' % (time_per_unit * 1e3, unit_name)

        else:

            formatted += ' %.0fus/%s' % (time_per_unit * 1e6, unit_name)

        return formatted

    def _estimate_step_duration(self, current: int=None, now: float=0.0) -> float:
        """Estimate the duration of a single step.
        Inputs:
        current: index of current step, default value is None (int)
        now: the current time, default value is 0.0 (float).
        """
        if current:
            
            if self._time_after_first_step is not None and current > 1:

                time_per_unit = (now - self._time_after_first_step) / (current - 1)
            else:

                time_per_unit = (now - self._start) / current

            if current == 1:

                self._time_after_first_step = now

            return time_per_unit

        else:

            return 0