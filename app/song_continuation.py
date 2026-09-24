"""Cut a saved song at a bar so YuE2 can continue it: Extend, Regenerate from here.

Every song keeps its semantic codec tokens (25 frames per second) and its ABC plan. The
semantic stage is a decoder-only language model whose prompt ends with the plan and
``MUSIC_START``; appending saved codec tokens continues that song. The plan is cut at the
same bar so the model can first continue the score and then the music.

Nominal bar times come from the score tempo. Generated audio follows the plan only
approximately, so bar times are scaled to the real duration when the two disagree.
Standard library only; usable from the web server for previews.
"""
import json
import re

from vendor import yue2_abc as abc_tools
import song_library

FRAMES_PER_SECOND = 25
CONTEXT = 24576
MIN_CONTINUATION = 200  # semantic tokens (8 s) the model may still write after a prompt
MAX_SCALE_DRIFT = 0.25
# Score labels (SheetSage2) → lyric tags the model saw in training.
TAGS = {
    'intro': 'Intro', 'verse': 'Verse', 'chorus': 'Chorus', 'pre-chorus': 'Pre-Chorus',
    'post-chorus': 'Post-Chorus', 'bridge': 'Bridge', 'interlude': 'Interlude', 'outro': 'Outro',
    'instrumental': 'Instrumental', 'rap': 'Rap', 'fade-out': 'Outro', 'preshot': 'Pre-Chorus',
    'intro and verse': 'Verse', 'solo': 'Interlude', 'break': 'Interlude',
}
VOCAL_FAMILIES = {'verse', 'chorus', 'pre-chorus', 'post-chorus', 'bridge', 'rap'}
SUNG_SECTION_NOTES = 8


def score_sections(abc):
    """Sections of a native two-voice ABC: label, bar span, sounding vocal notes and nominal seconds."""
    score = abc_tools.parse(abc)
    lines = abc.strip().splitlines()
    vocal = score.voices['Vocal']
    sections, labels, bar, i = [], [], 0, 8
    while i < len(lines):
        line = lines[i]
        if line.startswith('% '):
            labels.append(line[2:].strip().lower())
        elif line == 'V: Vocal':
            j = i + 1
            while lines[j].startswith(('M:', 'K:')):
                j += 1
            bars = 0
            for cell in lines[j][:-1].split('|'):
                rest = re.fullmatch(r'Z([2-4])?', cell.strip())
                bars += int(rest.group(1) or '1') if rest else 1
            if labels or not sections:
                sections.append({'labels': labels, 'first_bar': bar, 'bars': 0})
                labels = []
            sections[-1]['bars'] += bars
            bar += bars
        i += 1
    for k, s in enumerate(sections):
        start = vocal.bars[s['first_bar']][0]
        last = vocal.bars[s['first_bar'] + s['bars'] - 1]
        end = last[0] + last[1]
        s['notes'] = sum(1 for onset, _, _ in vocal.notes if start <= onset < end)
        s['nominal_start'] = float(start) * 60 / score.bpm
        s['nominal_end'] = float(end) * 60 / score.bpm
        label = s['labels'][-1] if s['labels'] else ('verse' if s['notes'] else 'intro')
        if label == 'silence':
            label = 'intro' if k == 0 else 'outro' if k == len(sections) - 1 else 'interlude'
        if label in ('irregular', 'loop'):
            label = 'verse' if s['notes'] >= SUNG_SECTION_NOTES else 'interlude'
        s['label'] = label
        s['tag'] = TAGS.get(label, label.title())
        s['index'] = k
        s['sung'] = label in VOCAL_FAMILIES or s['notes'] >= SUNG_SECTION_NOTES
    return sections, score



def frames(seconds):
    return int(round(seconds * FRAMES_PER_SECOND))


def bar_starts(score, scale=1.0):
    """Nominal start of every bar in seconds, scaled onto the audio."""
    return [float(start) * 60 / score.bpm * scale for start, _, _ in score.voices['Vocal'].bars]


def audio_scale(score, seconds):
    """Ratio that maps the plan's nominal duration onto the real song length, within reason."""
    nominal = float(score.voices['Vocal'].time) * 60 / score.bpm
    if not seconds or nominal <= 0:
        return 1.0
    ratio = seconds / nominal
    return ratio if abs(ratio - 1) <= MAX_SCALE_DRIFT else 1.0


def timeline(abc, seconds):
    """Bars and sections of a song for the picker; empty when the song has no usable plan."""
    result = {'seconds': round(seconds, 3), 'bpm': None, 'scale': 1.0, 'bars': [], 'sections': []}
    if not abc:
        return result
    try:
        sections, score = score_sections(abc)
    except (ValueError, KeyError, IndexError):
        return result
    scale = audio_scale(score, seconds)
    result.update(bpm=score.bpm, scale=round(scale, 4), bars=[round(t, 3) for t in bar_starts(score, scale)])
    result['sections'] = [{'label': s['label'], 'tag': s['tag'], 'sung': bool(s['sung']),
                           'start': round(s['nominal_start'] * scale, 3), 'end': round(s['nominal_end'] * scale, 3)}
                          for s in sections]
    return result


def expand_cells(music):
    """Bar cells of a music line with ``Z2``-style multi-bar rests expanded."""
    cells = []
    for cell in music[:-1].split('|'):
        rest = re.fullmatch(r'Z([2-4])?', cell.strip())
        cells.extend(['Z'] * int(rest.group(1) or '1') if rest else [cell])
    return cells


def join_cells(cells):
    """Music line from bar cells, writing runs of whole-bar rests as ``Z2``..``Z4`` like the model does."""
    out, i = [], 0
    while i < len(cells):
        if cells[i] == 'Z':
            n = 1
            while i + n < len(cells) and cells[i + n] == 'Z' and n < 4:
                n += 1
            out.append('Z' if n == 1 else f'Z{n}')
            i += n
        else:
            out.append(cells[i])
            i += 1
    return '|'.join(out) + '|'


def cut_score(abc, seconds, scale=1.0):
    """Return (abc_prefix, kept_bars, cut_seconds): the plan up to the bar starting at or before ``seconds``.

    The returned text has no end marker, so the model continues the score after it. Whole
    score when the cut lies in the last bar or beyond.
    """
    score = abc_tools.parse(abc)
    starts = bar_starts(score, scale)
    total = float(score.voices['Vocal'].time) * 60 / score.bpm * scale
    text = abc.strip() + '\n'
    if seconds >= total - 1e-6:
        return text, len(starts), total
    # The bar that starts at or before the time is regenerated; everything before it is kept.
    keep = max(1, sum(1 for t in starts if t <= seconds + 1e-6) - 1)
    lines = text.splitlines()
    out = lines[:8]
    done = 0
    i = 8
    while i < len(lines) and done < keep:
        if lines[i].startswith('% '):
            out.append(lines[i])
            i += 1
            continue
        group = []
        for name in ('Vocal', 'Ins'):
            if lines[i] != 'V: ' + name:
                raise ValueError('Unsupported score structure')
            j = i + 1
            while lines[j].startswith(('M:', 'K:')):
                j += 1
            group.append((lines[i:j], lines[j]))
            i = j + 1
        cells = [expand_cells(music) for _, music in group]
        n = min(len(cells[0]), keep - done)
        for (head, _), bars in zip(group, cells):
            selected = bars[:n]
            if n < len(bars) or done + n == keep:
                selected[-1] = re.sub(r'-\s*$', '', selected[-1])
            out += head + [join_cells(selected)]
        done += n
    while out and out[-1].startswith('% '):
        out.pop()
    result = '\n'.join(out) + '\n'
    after = abc_tools.parse(result)
    if len(after.voices['Vocal'].bars) != keep:
        raise ValueError('The score could not be cut at that bar')
    return result, keep, starts[keep]


def cut_codec(codec, seconds):
    n = min(len(codec), max(1, frames(seconds)))
    return list(codec[:n])


def budget(sampling, prefix_len, negative_len=0):
    """Sampling settings that fit the remaining context after a long prompt."""
    room = CONTEXT - max(prefix_len, negative_len)
    if room < MIN_CONTINUATION:
        raise ValueError('This song already fills the model context; cut it earlier to continue it.')
    result = dict(sampling)
    result['max_tokens'] = min(int(sampling['max_tokens']), room)
    result['min_tokens'] = min(int(sampling.get('min_tokens', 0)), result['max_tokens'])
    return result


def song_files(job, index):
    """Codec tokens, plan and settings of one saved version."""
    result, candidate = song_library.locate(job, index)
    prefix = (candidate or {}).get('prefix', '')
    if prefix not in ('', f'candidate-{index:02d}/'):
        raise ValueError('Invalid song version')
    folder = job / prefix if prefix else job
    tokens = folder / 'tokens.json'
    if not tokens.is_file():
        raise ValueError('This version has no saved music tokens to continue from.')
    codec = json.loads(tokens.read_text())
    if not isinstance(codec, list) or not codec:
        raise ValueError('The saved music tokens are empty.')
    settings_path = folder / 'settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else json.loads((job / 'request.json').read_text())
    score = folder / 'score.abc'
    abc = score.read_text() if score.is_file() else ((candidate or result).get('abc') or '')
    return {'codec': codec, 'abc': abc, 'settings': settings, 'seconds': len(codec) / FRAMES_PER_SECOND,
            'title': (candidate or {}).get('title') or settings.get('title') or ''}


def continuation_input(job, index, seconds):
    """What a continuation job needs: the cut codec and plan, snapped to a bar of the source."""
    song = song_files(job, index)
    cot = song['settings'].get('cot', 'full')
    total = song['seconds']
    seconds = total if seconds is None else max(0.0, min(float(seconds), total))
    abc_prefix, kept_bars, cut_seconds = '', 0, seconds
    if cot != 'off' and song['abc']:
        score = abc_tools.parse(song['abc'])
        scale = audio_scale(score, total)
        abc_prefix, kept_bars, cut_seconds = cut_score(song['abc'], seconds, scale)
        cut_seconds = min(cut_seconds, total)
    codec = cut_codec(song['codec'], cut_seconds)
    if len(codec) >= len(song['codec']) and seconds < total - 1 / FRAMES_PER_SECOND:
        codec = cut_codec(song['codec'], seconds)
    return {'codec': codec, 'abc_prefix': abc_prefix, 'cot': cot, 'kept_bars': kept_bars,
            'cut_seconds': round(len(codec) / FRAMES_PER_SECOND, 3),
            'source': {'job': job.name, 'candidate': index, 'seconds': round(total, 3), 'title': song['title']}}


def recording_continuation(tokens, seconds, title=''):
    """What a continuation job needs to extend an uploaded recording from its music tokens.

    ``tokens`` is the ``audio_tokens`` result cached for the recording. Recordings have no
    plan, so the cut is not snapped to bars and the song is generated score-free. The kept
    tokens are approximate and are rendered with the real-audio decoder (``real_audio``).
    """
    codec = tokens['codec']
    if not isinstance(codec, list) or not codec:
        raise ValueError('The recording has no music tokens.')
    total = len(codec) / FRAMES_PER_SECOND
    seconds = total if seconds is None else max(0.0, min(float(seconds), total))
    kept = cut_codec(codec, seconds)
    return {'codec': kept, 'abc_prefix': '', 'cot': 'off', 'kept_bars': 0, 'real_audio': True,
            'cut_seconds': round(len(kept) / FRAMES_PER_SECOND, 3),
            'source': {'recording': tokens['audio_id'], 'seconds': round(total, 3), 'title': title,
                       'tokenizer': tokens.get('tokenizer', '')}}
