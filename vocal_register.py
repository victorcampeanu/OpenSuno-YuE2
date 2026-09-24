"""Put a cover's transcribed melody in the register of the requested voice.

SheetSage2 writes the Vocal voice on a treble clef and transcribes a low male singer an
octave up; a melody that sat around E4 comes back centred on E5. The model sings the
pitches the score gives it, and three minutes of E5 outweigh one "Male lead vocal." tag
line, so the cover comes out with a female voice whatever the tag says.

This pass moves the Vocal voice by whole octaves: the sung median of the score decides against
the range of the requested voice, or an explicit choice is applied. Octave shifts keep every
chord symbol, key change, duration and tie valid; the Ins voice, the section labels and the bar
grid are untouched, and the result is re-parsed to prove it.
"""
from vendor import yue2_abc as abc_tools
import instrumental

CHOICES = ('auto', 'keep', 'down', 'down2', 'up')
# Duration-weighted median of the sung notes, MIDI. A lead melody centred at or above the high
# bound is out of reach for the voice and moves down an octave; at or below the low bound it
# moves up. Between them the register is left as heard; duets and unspecified voices never move.
REGISTERS = {'male': (47, 71), 'female': (55, 84)}    # male: below B2 / from B4; female: to G3 / from C6
NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')


def note_name(midi):
    return f'{NAMES[midi % 12]}{midi // 12 - 1}'


def median_pitch(abc):
    """Duration-weighted median MIDI pitch of the Vocal voice, or None when nothing is sung."""
    notes = abc_tools.parse(abc).voices['Vocal'].notes
    if not notes:
        return None
    total = sum(duration for _, _, duration in notes)
    covered = 0
    for _, pitch, duration in sorted(notes, key=lambda n: n[1]):
        covered += duration
        if covered * 2 >= total:
            return pitch
    return notes[-1][1]


def shift_line(music, octaves):
    """Move every sung note of one Vocal music line by whole octaves; rests, chords and keys stay."""
    def move(m):
        note = m.group('note')
        if note is None or note == 'z':
            return m.group(0)
        marks = m.group('oct')
        offset = (1 if note.islower() else 0) + marks.count("'") - marks.count(',') + octaves
        if offset >= 1:
            letter, marks = note.lower(), "'" * (offset - 1)
        else:
            letter, marks = note.upper(), ',' * -offset
        return (m.group('acc') or '') + letter + marks + m.group('duration') + m.group('tie')
    return abc_tools.TOKEN.sub(move, music)


def shift_octaves(abc, octaves):
    """Return the score with the Vocal voice moved by ``octaves``; verified against the source."""
    if not octaves:
        return abc
    source = abc_tools.parse(abc)
    lines = abc.strip().splitlines()
    for v, _ in instrumental.groups(lines):
        lines[v] = shift_line(lines[v], octaves)
    result = '\n'.join(lines) + '\n'
    after = abc_tools.parse(result)
    for name in abc_tools.VOICES:
        before_voice, after_voice = source.voices[name], after.voices[name]
        step = 12 * octaves if name == 'Vocal' else 0
        if [[t, p + step, d] for t, p, d in before_voice.notes] != [list(n) for n in after_voice.notes]:
            raise ValueError(f'Moving the {name} voice changed its notes')
        if (before_voice.bars, before_voice.chords, before_voice.keys) != (after_voice.bars, after_voice.chords, after_voice.keys):
            raise ValueError(f'Moving the {name} voice changed its bars, chords or keys')
    return result


def choose(median, voice, setting='auto'):
    """Octaves to move the Vocal voice: an explicit choice, or what the requested voice needs."""
    if setting not in CHOICES:
        raise ValueError(f'Unknown vocal range choice {setting!r}')
    if setting == 'down':
        return -1
    if setting == 'down2':   # a bass-baritone reading; three octaves would put the line under any singer
        return -2
    if setting == 'up':
        return 1
    if setting == 'keep' or median is None or voice not in REGISTERS:
        return 0
    low, high = REGISTERS[voice]
    if median >= high:
        return -1
    if median <= low:
        return 1
    return 0


def fit(abc, voice, setting='auto'):
    """Return (abc, octaves_moved, median_before) for a cover's transcribed score."""
    median = median_pitch(abc)
    octaves = choose(median, voice, setting)
    return shift_octaves(abc, octaves), octaves, median
