class EventOpt:

    '''
    Represents Event options for an event during an ode integration.

    name: Name of the event that the rest of the options are about. Must match the event name passed in an ode.

    max_events: Stop counting events after the specified number of them has been encoutered and stored. If max_events = 0, then all events will be ingored. If max_events = -1, all of them will be stored.

    terminate: If True, then the ode integration stops when max_events have been stored. If max_events = -1 or 0, this parameter is ignored.

    period: Events are only stored after every "period" encouters. If period=1, then all of them are stored. If period=2, then the first one is skipped, the second is stored, the thirs is skipped, etc...

    '''

    def __init__(self, name: str, max_events = -1, terminate = False, period=1):...
