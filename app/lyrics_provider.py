"""Read-only LRCLIB lyrics search."""
import json
import re
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

@router.get('/api/lyrics/search')
def search_lyrics(q: str = Query(min_length=2, max_length=200)):
    query = q.strip()
    if len(query) < 2:
        raise HTTPException(400, 'Enter a song title and artist.')
    request = Request('https://lrclib.net/api/search?' + urlencode({'q': query}),
        headers={'User-Agent': 'OpenSuno/1.0 (https://github.com/victorcampeanu/OpenSuno-YuE2)', 'Accept': 'application/json'})
    try:
        with urlopen(request, timeout=15) as response:
            records = json.load(response)
        if not isinstance(records, list):
            raise ValueError('Invalid provider response')
        results = []
        for record in records[:20]:
            lyrics = record.get('plainLyrics') or ''
            if not lyrics and record.get('syncedLyrics'):
                lyrics = re.sub(r'\[\d+:\d+(?:\.\d+)?\]', '', record['syncedLyrics']).strip()
            results.append({'id': record['id'], 'title': record.get('trackName', ''),
                'artist': record.get('artistName', ''), 'album': record.get('albumName', ''),
                'duration': record.get('duration', 0), 'instrumental': bool(record.get('instrumental')),
                'lyrics': lyrics})
        return {'results': results, 'provider': 'LRCLIB'}
    except HTTPError as error:
        if error.code == 429:
            raise HTTPException(429, 'Lyrics provider is busy. Wait a moment and search again.') from error
        raise HTTPException(502, 'Lyrics provider is unavailable. Try again shortly.') from error
    except (URLError, TimeoutError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(502, 'Could not search lyrics. Check your internet connection and try again.') from error
