from contextlib import contextmanager
from threading import Event
from typing import List, Optional, Union


def set_all(ev_list: List[Event]):
    for ev in ev_list:
        ev.set()


def clear_all(ev_list: List[Event]):
    for ev in ev_list:
        ev.clear()


@contextmanager
def catch_loop_errors(
    *,
    running_event: Optional[Event] = None,
    event_to_be_set: Union[Event, List[Event]] = [],
    event_to_be_cleared: Union[Event, List[Event]] = [],
):
    """
    Context manager to be used in the processing loops of nodes.
    It is used to catch exceptions and interrrupt the application as best as possible.

    Args:
        *
        running_event: event that, if cleared, stops the system - will only be cleared
            if an exception is raised
        event_to_be_set: event or list of; they will be SET when the loop is interrupted
            or ends.
        event_to_be_cleared: event or list of; they will be CLEARED when the loop is
            interrupted or ends.
    """
    if isinstance(event_to_be_set, Event):
        event_to_be_set = [event_to_be_set]
    if isinstance(event_to_be_cleared, Event):
        event_to_be_cleared = [event_to_be_cleared]

    try:
        yield
    except KeyboardInterrupt:
        if running_event:
            running_event.clear()
        set_all(event_to_be_set)
        clear_all(event_to_be_cleared)
    except Exception as e:
        if running_event:
            running_event.clear()
        set_all(event_to_be_set)
        clear_all(event_to_be_cleared)
        raise e
    else:
        set_all(event_to_be_set)
        clear_all(event_to_be_cleared)
