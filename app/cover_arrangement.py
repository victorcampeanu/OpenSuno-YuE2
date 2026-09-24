"""Arrange a transcribed cover score with the model.

SheetSage2 hears one lead line: the sung melody, and the instrumental lead in the gaps
between verses. With "Preserve: Melody" it writes no chord symbols at all. YuE2's own plans
carry a chord on nearly every bar and give the Ins voice something to play wherever nobody
sings, and the semantic stage renders what the score says. A bare transcription therefore
comes out as a bare arrangement: a lone line under the voice.

This pass keeps everything the recording gave us (the sung notes, the heard instrumental
lines, the section labels, meter and key changes, the bar grid) and lets the model write the
rest in the requested style:

* a chord symbol at the start of every bar of the Vocal voice that has none, and
* an instrumental line for sections where nothing melodic was heard at all.

The model writes into its own score format token by token; the transcribed material is
forced between its choices, so it harmonises exactly the notes that will be sung. Any
``writer`` with a ``text`` attribute, ``commit(text)`` (append forced text) and
``propose(max_tokens, stop, first=None)`` (sample a continuation of the committed text
without committing it; ``first`` is a regex the first token's text should match, which a
writer may ignore) works; the MLX and CUDA engines each provide one.

The model's choices are checked against the music they accompany (``in_tune``). Left alone,
sampling drifts: a chord whose root is a semitone off the key (the BPE spells ``Ebm`` as
``E`` + ``bm``, and a stray closing quote leaves ``E``), a major chord under a sung minor
third, and once one wrong chord is in the score the model copies it for the rest of the
song. Every proposed chord must therefore have its root in the key and agree with the
notes sung in its bar; a proposal that does not is replaced by the closest chord that
does, so the bar keeps a chord and the score never teaches the model a wrong one. The
transcription's key label is only trusted for its tonic, not its mode (a verse in E-flat
minor under a chorus in E-flat major is common and SheetSage2 picks one), so roots from
both parallel modes are allowed and the mode is decided bar by bar from what is sung. An
instrumental line the model writes is accepted only when it plays the chords it sits on.
"""
import re
from dataclasses import dataclass, field

from vendor import yue2_abc as abc_tools

CHORD_TOKENS = 8          # a chord name and its closing quote are 2–5 tokens; more means the model went elsewhere
CHORD_ATTEMPTS = 2
OPENING_QUOTE = '"[A-G]'  # token texts allowed to open a chord at a line start (regex, full match)
INS_TOKENS_PER_BAR = 48   # a dense 4/4 bar in L:1/16 is ~35 tokens
INS_ATTEMPTS = 3
FULL_REST = re.compile(r'Z([2-4])?')
DURATIONS = sorted(abc_tools.DURATIONS, reverse=True)

# ── Harmony ──────────────────────────────────────────────────────────────────
PITCH_CLASS = dict(zip('CDEFGAB', (0, 2, 4, 5, 7, 9, 11)))
ACCIDENTAL = {'': 0, 'b': -1, 'bb': -2, '#': 1, '##': 2}
FLAT_NAMES = ('C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B')
SHARP_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
INTERVALS = {'': (0, 4, 7), 'm': (0, 3, 7), 'dim': (0, 3, 6), 'aug': (0, 4, 8), '7': (0, 4, 7, 10),
             'maj7': (0, 4, 7, 11), 'm7': (0, 3, 7, 10), 'dim7': (0, 3, 6, 9), 'm7b5': (0, 3, 6, 10),
             'sus4': (0, 5, 7), 'sus2': (0, 2, 7), '6': (0, 4, 7, 9), 'm6': (0, 3, 7, 9),
             '7sus4': (0, 5, 7, 10), 'm(maj7)': (0, 3, 7, 11)}
CHORD_PARTS = re.compile(r'(?P<root>[A-G](?:bb|##|b|#)?)(?P<quality>[^/]*)(?:/(?P<bass>[A-G](?:bb|##|b|#)?))?')
# Scale degrees (semitones above the tonic) of the parallel major and minor together: everything
# but the flat second and the sharp fourth, the two roots that are never in tune with either mode.
KEY_DEGREES = frozenset({0, 2, 3, 4, 5, 7, 8, 9, 10, 11})
MIN_CHORD_FIT = 1 / 3     # sung time on chord tones needed to accept a chord for a bar with sung notes
MIN_INS_FIT = 0.5         # note time on chord tones needed to accept an instrumental line
MAX_INS_OUT_OF_KEY = 0.1  # note time an instrumental line may spend outside the key


def pitch_class(name):
    """``'Eb'`` → 3; also accepts key names such as ``'Ebm'``."""
    m = re.match(r'([A-G])(bb|##|b|#)?', name)
    return (PITCH_CLASS[m.group(1)] + ACCIDENTAL[m.group(2) or '']) % 12


def chord_tones(chord):
    """``(root pitch class, third in semitones or None for sus chords, {pitch classes})`` of a chord name."""
    m = CHORD_PARTS.fullmatch(chord)
    root = pitch_class(m.group('root'))
    intervals = INTERVALS[m.group('quality')]
    tones = {(root + i) % 12 for i in intervals}
    if m.group('bass'):
        tones.add(pitch_class(m.group('bass')))
    third = 3 if 3 in intervals else 4 if 4 in intervals else None
    return root, third, tones


def spell(pc, key):
    """The chord-root spelling of a pitch class that suits the key (flats in flat keys and C, sharps in sharp keys)."""
    return (SHARP_NAMES if abc_tools.KEYS.get(key, 0) > 0 else FLAT_NAMES)[pc % 12]


def chord_fit(chord, melody, key):
    """How well a chord suits the sung notes of a bar: the share of sung time spent on chord tones.

    ``melody`` maps pitch classes to sung time. Returns None when the chord is out of the
    question: one of its tones is outside the key (either parallel mode; this also rules
    out major triads on the leading tone and secondary dominants), the melody sings the
    other third more than the chord's own (a major chord under a minor melody or vice
    versa), or it dwells on notes a semitone above chord tones at least as long as on the
    chord tones themselves (E-natural over an A-flat chord; a passing note is fine).
    A bar with nothing sung fits any chord in the key.
    """
    root, third, tones = chord_tones(chord)
    tonic = pitch_class(key)
    if any((pc - tonic) % 12 not in KEY_DEGREES for pc in tones):
        return None
    total = sum(melody.values())
    if not total:
        return 1.0
    if third is not None and melody.get((root + 7 - third) % 12, 0) > melody.get((root + third) % 12, 0):
        return None
    on_chord = sum(w for pc, w in melody.items() if pc in tones)
    clash = sum(w for pc, w in melody.items() if (pc - 1) % 12 in tones and pc not in tones)
    if clash and clash >= on_chord:
        return None
    return on_chord / total


def acceptable(chord, melody, key):
    fit = chord_fit(chord, melody, key)
    return fit is not None and fit >= MIN_CHORD_FIT


def scale(key):
    """Pitch classes of the key label's own scale; minor keys include the raised seventh (their major dominant)."""
    tonic = pitch_class(key)
    return {(tonic + d) % 12 for d in ((0, 2, 3, 5, 7, 8, 10, 11) if key.endswith('m') else (0, 2, 4, 5, 7, 9, 11))}


def diatonic(chord, key):
    return chord_tones(chord)[2] <= scale(key)


def best_chord(melody, key, previous=None, strict=False):
    """The chord to write when the model's proposal is out of tune.

    Keeps the previous chord when it still fits (harmony moves when the melody asks it to),
    otherwise the major or minor triad in the key with the most sung time on its tones,
    preferring diatonic chords, then the tonic, then the key's own mode, then nearness to
    the previous root. ``strict`` considers diatonic chords only.
    """
    if previous and acceptable(previous, melody, key) and (not strict or diatonic(previous, key)):
        return previous
    tonic = pitch_class(key)
    own_quality = 'm' if key.endswith('m') else ''
    prev_root = chord_tones(previous)[0] if previous else tonic
    candidates = []
    for degree in sorted(KEY_DEGREES):
        root = (tonic + degree) % 12
        for quality in ('', 'm'):
            chord = spell(root, key) + quality
            fit = chord_fit(chord, melody, key)
            if fit is None or (strict and not diatonic(chord, key)):
                continue
            distance = min((root - prev_root) % 12, (prev_root - root) % 12)
            candidates.append((fit, diatonic(chord, key), root == tonic, quality == own_quality, -distance, chord))
    if not candidates:
        return spell(tonic, key) + own_quality
    return max(candidates)[-1]


def settle(chord, melody, key, previous):
    """An accepted chord, spelled for the key; a chord borrowed from outside the key's own scale is kept
    only when it fits the bar better than the best diatonic chord (``G#`` under D#–G#–C# in E was ``G#m``).
    Returns ``(chord, changed)``."""
    chord = respell(chord, key)
    if diatonic(chord, key):
        return chord, False
    alternatives = [best_chord(melody, key, None, strict=True)]
    if previous and diatonic(previous, key) and acceptable(previous, melody, key):
        alternatives.insert(0, previous)  # ties go to continuity
    alternative = max(alternatives, key=lambda c: chord_fit(c, melody, key) or -1)
    alternative_fit = chord_fit(alternative, melody, key)
    if alternative_fit is not None and alternative_fit >= chord_fit(chord, melody, key):
        return alternative, True
    return chord, False


def respell(chord, key, shift=0):
    """The chord with its root (moved ``shift`` semitones) and bass spelled for the key: ``B#`` in E is ``C``."""
    m = CHORD_PARTS.fullmatch(chord)
    bass = '/' + spell(pitch_class(m.group('bass')), key) if m.group('bass') else ''
    return spell(pitch_class(m.group('root')) + shift, key) + m.group('quality') + bass


def root_variants(chord, key):
    """The same chord a semitone lower and higher, spelled for the key: the model's ``E`` in E-flat was ``Eb`` or ``Ebm``."""
    return [respell(chord, key, -1), respell(chord, key, 1)]


def weights_by_bar(notes, bars):
    """Per bar ``(start, length, meter)``, the sounding time (quarter notes) per pitch class; tied notes count where they sound."""
    result = [{} for _ in bars]
    for start, pitch, duration in notes:
        end = start + duration
        for i, (bar_start, length, _) in enumerate(bars):
            overlap = min(end, bar_start + length) - max(start, bar_start)
            if overlap > 0:
                weights = result[i]
                weights[pitch % 12] = weights.get(pitch % 12, 0) + overlap
    return result


def melody_by_bar(score):
    """For each bar, the notes a chord must agree with: what is sung, or where nothing is sung, the
    instrumental line heard in the recording (so an intro riff gets its own harmony, not the tonic held under it)."""
    bars = score.voices['Vocal'].bars
    sung = weights_by_bar(score.voices['Vocal'].notes, bars)
    heard = weights_by_bar(score.voices['Ins'].notes, bars)
    return [s or h for s, h in zip(sung, heard)]


def ins_fit(voice, chords, key):
    """``(chord-tone share, out-of-key share)`` of a parsed instrumental voice against the chords of its bars."""
    tonic = pitch_class(key)
    on_chord = under_chord = out_of_key = total = 0
    for start, pitch, duration in voice.notes:
        index = max(i for i, (bar_start, _, _) in enumerate(voice.bars) if bar_start <= start)
        chord = chords[index] if index < len(chords) else None
        total += duration
        if chord is not None:
            under_chord += duration
            if pitch % 12 in chord_tones(chord)[2]:
                on_chord += duration
        if (pitch - tonic) % 12 not in KEY_DEGREES:
            out_of_key += duration
    return (on_chord / under_chord if under_chord else 1.0), (out_of_key / total if total else 0.0)


class ArrangementError(ValueError):
    """The score could not be arranged; callers keep the transcription instead."""


@dataclass
class Arrangement:
    abc: str
    bars: int = 0
    chords: int = 0
    kept_chords: int = 0
    corrected_chords: int = 0   # proposals that were out of tune and replaced by a fitting chord
    ins_lines: int = 0
    ins_failed: int = 0
    ins_rejected: int = 0       # well-formed instrumental lines refused for not playing their chords
    notes: list = field(default_factory=list)


def rest_bar(meter, unit_denominator):
    """A full bar of explicit rests, e.g. ``z16`` for 4/4 in L:1/16, so a chord can sit on it."""
    n, d = meter
    units = n * unit_denominator
    if units % d:
        raise ArrangementError(f'Meter {n}/{d} does not divide into L:1/{unit_denominator} units')
    units //= d
    parts = []
    while units:
        length = next((x for x in DURATIONS if x <= units), None)
        if length is None:
            raise ArrangementError(f'Cannot write a {units}-unit rest')
        parts.append(f'z{length}')
        units -= length
    return ''.join(parts)


def bars_of(music):
    """Bar bodies of a music line, full-measure ``Z`` rests kept as a single ``Z`` each."""
    if not music.endswith('|'):
        raise ArrangementError('Music line must end with a barline')
    bars = []
    for cell in music[:-1].split('|'):
        cell = cell.strip()
        if not cell:
            raise ArrangementError('Empty measure')
        rest = FULL_REST.fullmatch(cell)
        bars.extend(['Z'] * int(rest.group(1) or '1') if rest else [cell])
    return bars


def is_silent(music):
    return not any(m.group('note') not in (None, 'z') for m in abc_tools.TOKEN.finditer(music))


def read_chord(text):
    """The chord name the model wrote after a forced opening quote, or None.

    The closing quote usually arrives merged with the first note (``"d``); anything after it
    is the model's guess at the melody and is discarded in favour of the transcription.
    """
    if '"' not in text:
        return None
    chord = text[:text.index('"')]
    return chord if abc_tools.CHORD.fullmatch(chord) else None


def uncommit(writer, text):
    if not writer.text.endswith(text):
        raise ArrangementError('Writer text diverged from the arrangement')
    writer.text = writer.text[:-len(text)]


def tune(chord, melody, key, previous):
    """The chord to write for a proposal: the proposal itself when it fits the bar, else the
    nearest fitting respelling of its root, else the best fitting chord. ``melody`` None skips the checks.
    Accepted chords are settled: spelled for the key (``B#`` → ``C``) so the score never teaches the
    model odd names, and borrowed chords kept only when the melody asks for them."""
    if melody is None:
        return chord, False
    if acceptable(chord, melody, key):
        return settle(chord, melody, key, previous)
    for variant in root_variants(chord, key):
        if acceptable(variant, melody, key):
            return settle(variant, melody, key, previous)[0], True
    return best_chord(melody, key, previous), True


def propose_chord(writer, barline, melody=None, key='C', previous=None):
    """Open a chord symbol at the bar start and let the model name it.

    The model's own plans put a chord on nearly every bar, so the model only chooses which
    chord, not whether to write one: asked that instead, a model that skips the first bar
    reads the score as chord-free and never adds any. The quote is opened the way the BPE
    would have written it. After a barline that is the merged ``|"`` token, which is
    committed as text. At a line start the quote merges with the chord's letter (``"B``),
    so the first sampled token is constrained to those tokens instead; a lone ``"`` there
    is something the model never saw and it answers with one odd chord repeated for the
    whole song. Writers that cannot constrain sampling may then propose rests first, and
    the bar stays unannotated. An unusable name is retried once.

    With ``melody`` (the bar's sung time per pitch class) the name is checked against the
    bar and the key and corrected when out of tune; see ``tune``. Returns
    ``(chord or None, corrected)``.
    """
    if barline:
        writer.commit('|"')
        for _ in range(CHORD_ATTEMPTS):
            chord = read_chord(writer.propose(CHORD_TOKENS, lambda t: '"' in t))
            if chord:
                chord, corrected = tune(chord, melody, key, previous)
                writer.commit(chord + '"')
                return chord, corrected
        uncommit(writer, '|"')
        writer.commit('|')
        return None, False
    for _ in range(CHORD_ATTEMPTS):
        text = writer.propose(CHORD_TOKENS, lambda t: not t.startswith('"') or t.count('"') >= 2, first=OPENING_QUOTE)
        chord = read_chord(text[1:]) if text.startswith('"') else None
        if chord:
            chord, corrected = tune(chord, melody, key, previous)
            writer.commit(f'"{chord}"')
            return chord, corrected
    return None, False


def parse_ins_line(line, bars, header, meter, key):
    """The parsed Ins voice of an instrumental line the model wrote for a silent section, checked as its own tiny score; None if unusable."""
    if not line or not line.endswith('|') or '"' in line or '[' in line:
        return None
    try:
        if len(bars_of(line)) != bars:
            return None
        probe = list(header)
        probe[2] = f'M:{meter[0]}/{meter[1]}'
        probe[7] = f'K:{key}'
        probe += ['V: Vocal', 'Z' + (str(bars) if bars > 1 else '') + '|', 'V: Ins', line]
        return abc_tools.parse('\n'.join(probe) + '\n').voices['Ins']
    except (ArrangementError, abc_tools.AbcError):
        return None


def valid_ins_line(line, bars, header, meter, key):
    return parse_ins_line(line, bars, header, meter, key) is not None


def in_tune_ins_line(voice, chords, key):
    """A written instrumental line plays the chords of its bars and stays in the key."""
    on_chord, out_of_key = ins_fit(voice, chords, key)
    return on_chord >= MIN_INS_FIT and out_of_key <= MAX_INS_OUT_OF_KEY


def propose_ins_line(writer, bars, header, meter, key, chords=None):
    """``(line or None, rejected)``: ``rejected`` counts well-formed lines refused for being out of tune
    (only with ``chords``, the chord of each bar of the section)."""
    rejected = 0
    for _ in range(INS_ATTEMPTS):
        text = writer.propose(INS_TOKENS_PER_BAR * bars + CHORD_TOKENS, lambda t: '\n' in t)
        line = text.split('\n', 1)[0].strip()
        voice = parse_ins_line(line, bars, header, meter, key)
        if voice is None:
            continue
        if chords is not None and not in_tune_ins_line(voice, chords, key):
            rejected += 1
            continue
        return line, rejected
    return None, rejected


def bar_chord(body):
    """The chord symbol a transcribed bar already carries, or None."""
    m = re.search(r'"([^"]*)"', body)
    return m.group(1) if m else None


def arrange(abc, writer, on_progress=None, fill_instrumental=True, in_tune=True):
    """Return the transcription with the model's chords and instrumental lines written in.

    ``in_tune`` checks every model choice against the sung notes, the key and the chords
    (see the module docstring); off, well-formed proposals are taken as written.
    Raises ArrangementError when the score is not in the two-voice dialect or the result
    fails to parse; the sung notes and the bar grid are verified unchanged.
    """
    try:
        before = abc_tools.parse(abc)
    except abc_tools.AbcError as e:
        raise ArrangementError(str(e)) from e
    lines = abc.strip().splitlines()
    header = lines[:8]
    unit_denominator = int(header[3][4:])
    meter = abc_tools.meter_value(header[2][2:])
    key = header[7][2:]
    total = len(before.voices['Vocal'].bars)
    melodies = melody_by_bar(before) if in_tune else [None] * total
    previous = None
    result = Arrangement(abc='')
    writer.commit('\n'.join(header) + '\n')

    def take_fields(i, apply):
        nonlocal meter, key
        fields = []
        while i < len(lines) and lines[i].startswith(('M:', 'K:')):
            fields.append(lines[i])
            if apply:
                if lines[i].startswith('M:'):
                    meter = abc_tools.meter_value(lines[i][2:])
                else:
                    key = lines[i][2:]
            i += 1
        return fields, i

    i = 8
    while i < len(lines):
        while lines[i].startswith('% '):
            writer.commit(lines[i] + '\n')
            i += 1
        if lines[i] != 'V: Vocal':
            raise ArrangementError('Unsupported score structure')
        fields, i = take_fields(i + 1, apply=True)
        writer.commit('V: Vocal\n' + ''.join(f + '\n' for f in fields))
        vocal = lines[i]
        i += 1
        if i >= len(lines) or lines[i] != 'V: Ins':
            raise ArrangementError('Unsupported score structure')
        ins_fields, i = take_fields(i + 1, apply=False)
        ins = lines[i]
        i += 1

        bars = bars_of(vocal)
        rest = rest_bar(meter, unit_denominator)
        section_chords = []
        for index, body in enumerate(bars):
            if body == 'Z':
                body = rest
            barline = '' if index == 0 else '|'
            if '"' in body:
                result.kept_chords += 1
                writer.commit(barline)
                chord = bar_chord(body)
            else:
                chord, corrected = propose_chord(writer, barline, melodies[result.bars], key, previous)
                if chord:
                    result.chords += 1
                    result.corrected_chords += corrected
            section_chords.append(chord)
            previous = chord or previous
            writer.commit(body)
            result.bars += 1
            if on_progress is not None:
                on_progress(result.bars, total)
        writer.commit('|\nV: Ins\n' + ''.join(f + '\n' for f in ins_fields))
        if fill_instrumental and is_silent(vocal) and is_silent(ins):
            written, rejected = propose_ins_line(writer, len(bars), header, meter, key, section_chords if in_tune else None)
            result.ins_rejected += rejected
            if written:
                result.ins_lines += 1
                ins = written
            else:
                result.ins_failed += 1
        writer.commit(ins + '\n')

    text = writer.text
    try:
        after = abc_tools.parse(text)
    except abc_tools.AbcError as e:
        raise ArrangementError(f'The arranged score does not parse: {e}') from e
    for name in abc_tools.VOICES:
        if after.voices[name].bars != before.voices[name].bars:
            raise ArrangementError(f'The {name} bar grid changed')
    if after.voices['Vocal'].notes != before.voices['Vocal'].notes:
        raise ArrangementError('The sung melody changed')
    result.abc = text
    return result
