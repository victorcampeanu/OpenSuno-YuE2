"""Native instrumental mode: silence the Vocal voice of a YuE2 score and keep its melody.

The semantic stage follows the plan. A score whose Vocal voice is all rests is the
model's own "nobody sings here" signal, which is far more reliable than asking for
"no vocals" in the style text. The sung melody is moved into the Ins voice so the
hook survives; chord symbols stay on the (now silent) Vocal voice where the dialect
requires them.
"""
import re

from vendor import yue2_abc as abc_tools

TAG_LINE = re.compile(r'^\s*\[([^\[\]\n]{1,40})\]\s*$')
DEFAULT_STRUCTURE = '\n\n'.join('[' + t + ']' for t in
                                ('Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Verse', 'Pre-Chorus', 'Chorus', 'Bridge', 'Chorus', 'Outro')) + '\n'
STYLE_LINE = 'Instrumental.'



def has_notes(music):
    return any(m.group('note') not in (None, 'z') for m in abc_tools.TOKEN.finditer(music))


def move_vocal(vocal_music, ins_music):
    """Return (vocal_line, ins_line, moved) with the vocal melody played by the instrument.

    A sung line becomes rests of the same durations; its chords and inline key changes
    stay. The instrument takes the vocal melody with the quoted chords removed. When the
    vocal line is already silent both lines are returned unchanged.
    """
    if not has_notes(vocal_music):
        return vocal_music, ins_music, False

    def rest(m):
        if m.group('chord') is not None or m.group('key') is not None:
            return m.group(0)
        return 'z' + m.group('duration')
    rests = abc_tools.TOKEN.sub(rest, vocal_music)
    motif = re.sub(r'"[^"\n]*"', '', vocal_music)
    return rests, motif, True


def groups(lines, start=8):
    """Yield (vocal_index, ins_index) of the music lines of each Vocal/Ins group."""
    i = start
    while i < len(lines):
        if lines[i].startswith('% '):
            i += 1
            continue
        if lines[i] != 'V: Vocal':
            raise ValueError('Unsupported score structure')
        v = i + 1
        while lines[v].startswith(('M:', 'K:')):
            v += 1
        k = v + 1
        if lines[k] != 'V: Ins':
            raise ValueError('Unsupported score structure')
        n = k + 1
        while lines[n].startswith(('M:', 'K:')):
            n += 1
        yield v, n
        i = n + 1


def bar_count(music):
    return sum(int(m.group(1) or '1') if (m := re.fullmatch(r'Z([2-4])?', cell.strip())) else 1
               for cell in music[:-1].split('|'))


def complete_score(text):
    """Drop a trailing Vocal/Ins group that the planning token cap cut short."""
    lines = text.strip().splitlines()
    last = None
    try:
        for v, n in groups(lines):
            if not (lines[v].endswith('|') and lines[n].endswith('|')):
                break
            last = n
    except (ValueError, IndexError):
        pass
    if last is None or last == len(lines) - 1:
        return text
    return '\n'.join(lines[:last + 1]) + '\n'


HOOK_SECTION = re.compile(r'chorus|hook|refrain', re.I)
PRE_HOOK = re.compile(r'pre', re.I)


def is_hook(label):
    """Chorus / hook / refrain sections carry the tune; a pre-chorus is a build, not the hook."""
    return bool(label) and bool(HOOK_SECTION.search(label)) and not PRE_HOOK.search(label)


def section_of(lines, index):
    """The ``% label`` in force at a music line, or '' before the first label."""
    for line in reversed(lines[:index]):
        if line.startswith('% '):
            return line[2:].strip()
    return ''


def instrumental_score(text, hook_only=False):
    """Return (abc, moved_bars, freed_bars): the score with a silent Vocal voice and its melody in Ins.

    With ``hook_only`` the instrument takes the melody only in chorus / hook sections; in the other
    sections the sung line is simply silenced, leaving those bars to the chords, so the model arranges
    them fully instead of shadowing a lead line for the whole song. Scores without section labels
    keep the melody everywhere.
    """
    try:
        source = abc_tools.parse(text)
    except ValueError:
        text = complete_score(text)
        source = abc_tools.parse(text)
    lines = text.strip().splitlines()
    labelled = any(line.startswith('% ') for line in lines)
    moved_bars = freed_bars = 0
    previous_moved = None
    previous_ins = None
    for v, n in groups(lines):
        vocal, ins, moved = move_vocal(lines[v], lines[n])
        if moved and hook_only and labelled and not is_hook(section_of(lines, v)):
            ins, moved = lines[n], False
            freed_bars += bar_count(lines[v])
        elif moved:
            moved_bars += bar_count(lines[v])
        # A tie may not run between a kept instrument phrase and a moved vocal phrase.
        if previous_ins is not None and previous_moved != moved:
            lines[previous_ins] = re.sub(r'-(?=\s*\|\s*$)', '', lines[previous_ins])
        lines[v], lines[n] = vocal, ins
        previous_moved, previous_ins = moved, n
    result = '\n'.join(lines) + '\n'
    after = abc_tools.parse(result)
    if after.voices['Vocal'].notes:
        raise ValueError('The vocal voice was not silenced')
    if after.voices['Vocal'].bars != source.voices['Vocal'].bars:
        raise ValueError('Silencing the vocal voice changed the bar grid')
    return result, moved_bars, freed_bars


def structure_only(lyrics):
    """Keep only section tags from the lyrics; fall back to a conventional song structure."""
    tags = [m.group(1).strip() for m in (TAG_LINE.match(line) for line in (lyrics or '').splitlines()) if m]
    if not tags:
        return DEFAULT_STRUCTURE
    return '\n\n'.join('[' + t + ']' for t in tags) + '\n'


SCORE_LABEL = re.compile(r'^%\s*([A-Za-z][A-Za-z -]{0,30})\s*$')


def score_sections(abc):
    """Section names of a transcribed score, from its ``% label`` lines, as tags: ``['Intro', 'Verse', ...]``."""
    return [m.group(1).strip().title() for m in (SCORE_LABEL.match(line) for line in (abc or '').splitlines()) if m]


def cover_structure(abc, lyrics):
    """The structure of an instrumental cover: the recording's own sections, so the plan mirrors the
    transcribed score bar for bar; tags typed by the user only matter when the score has no labels."""
    tags = score_sections(abc)
    return '\n\n'.join('[' + t + ']' for t in tags) + '\n' if tags else structure_only(lyrics)


def instrumental_style(style):
    return (style or '').rstrip() + '\n' + STYLE_LINE
