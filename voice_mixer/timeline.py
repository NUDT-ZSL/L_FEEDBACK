"""Convert ordered events into sample-accurate render fragments."""

from .errors import E_EVENT, E_TICK_ORDER
from .models import PttBurst, RenderFragment, Timeline

PTT_DURATION_MS = 40.0


class TimelineError(Exception):
    """Collects structured event-level timeline errors."""

    def __init__(self, errors):
        super().__init__("timeline validation failed")
        self.errors = errors


def tick_to_sample(tick, sample_rate, tick_rate):
    return (tick * sample_rate) // tick_rate


class _SpeakerState:
    def __init__(self, speaker):
        self.speaker = speaker
        self.channel = speaker.bound_channel
        self.speaking = False
        self.blocked = False
        self.start_sample = 0
        self.source_offset = 0


def build_timeline(project):
    errors = []
    timeline = Timeline()
    states = {sid: _SpeakerState(sp) for sid, sp in project.speakers.items()}
    sample_rate = project.sample_rate
    tick_rate = project.tick_rate

    last_tick = None
    for event in project.events:
        if last_tick is not None and event.tick < last_tick:
            errors.append(
                {
                    "code": E_TICK_ORDER,
                    "message": "events are not ordered by tick",
                    "event_index": event.index,
                    "path": f"events[{event.index}].tick",
                }
            )
        last_tick = event.tick

        state = states[event.speaker]
        at = tick_to_sample(event.tick, sample_rate, tick_rate)

        if event.type == "start":
            if state.speaking:
                errors.append(
                    _event_error(
                        event.index,
                        "nested 'start' while already speaking; previous segment is closed at this tick",
                    )
                )
                _close_fragment(state, at, timeline, errors, event.index)
            state.speaking = True
            state.start_sample = at
        elif event.type == "end":
            if not state.speaking:
                errors.append(
                    _event_error(event.index, "'end' event has no matching 'start'")
                )
            else:
                _close_fragment(state, at, timeline, errors, event.index)
                state.speaking = False
        elif event.type == "switch":
            new_channel = event.channel or state.speaker.bound_channel
            if state.speaking:
                _close_fragment(state, at, timeline, errors, event.index)
                state.start_sample = at
            state.channel = new_channel
        elif event.type == "block":
            if state.speaking and not state.blocked:
                _close_fragment(state, at, timeline, errors, event.index)
                state.start_sample = at
            state.blocked = True
        elif event.type == "unblock":
            if state.speaking and state.blocked:
                _close_fragment(state, at, timeline, errors, event.index)
                state.start_sample = at
            state.blocked = False
        elif event.type == "ptt":
            length = max(1, int(round(PTT_DURATION_MS * sample_rate / 1000.0)))
            timeline.ptt_bursts.append(
                PttBurst(
                    speaker_id=event.speaker,
                    channel_id=state.channel,
                    start_sample=at,
                    length_samples=length,
                    blocked=state.blocked,
                )
            )

    final_tick = max((ev.tick for ev in project.events), default=0)
    final_sample = tick_to_sample(final_tick, sample_rate, tick_rate)
    for sid, state in states.items():
        if state.speaking:
            errors.append(
                {
                    "code": E_EVENT,
                    "message": f"speaker '{sid}' has a 'start' without a matching 'end'; closed at final tick",
                    "path": f"speakers.{sid}",
                }
            )
            _close_fragment(state, final_sample, timeline, errors, None)

    timeline.fragments.sort(key=lambda frag: (frag.start_sample, frag.speaker_id))
    timeline.ptt_bursts.sort(key=lambda burst: (burst.start_sample, burst.speaker_id))
    voice_ends = [frag.end_sample for frag in timeline.fragments]
    ptt_ends = [burst.start_sample + burst.length_samples for burst in timeline.ptt_bursts]
    timeline.last_voice_end = max(voice_ends + ptt_ends + [0])

    if errors:
        raise TimelineError(errors)
    return timeline


def _close_fragment(state, end_sample, timeline, errors, event_index):
    start = state.start_sample
    length = end_sample - start
    if length < 0:
        errors.append(
            _event_error(event_index, "segment ends before it starts (negative length)")
        )
        return
    if length == 0:
        errors.append(
            _event_error(event_index, "zero-length segment at identical start/end sample")
        )
    else:
        timeline.fragments.append(
            RenderFragment(
                speaker_id=state.speaker.speaker_id,
                channel_id=state.channel,
                start_sample=start,
                end_sample=end_sample,
                blocked=state.blocked,
                source_offset=state.source_offset,
            )
        )
        state.source_offset += length


def _event_error(index, message):
    obj = {"code": E_EVENT, "message": message}
    if index is not None:
        obj["event_index"] = index
        obj["path"] = f"events[{index}]"
    return obj
