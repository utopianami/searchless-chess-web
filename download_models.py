"""Download and verify all five official archives, retaining ZIP backups."""
import base64
import hashlib
from pathlib import Path
import urllib.request
import zipfile

from model_specs import MODEL_SPECS

ROOT = Path(__file__).resolve().parent / 'searchless_chess' / 'checkpoints'


def digest(path):
    with path.open('rb') as stream:
        return base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()


def main():
    archives = ROOT / 'archives'
    archives.mkdir(parents=True, exist_ok=True)
    for model in MODEL_SPECS:
        url = f'https://storage.googleapis.com/searchless_chess/checkpoints/{model}.zip'
        archive = archives / f'{model}.zip'
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
            hashes = ','.join(response.headers.get_all('x-goog-hash', []))
            expected = next(part.strip().split('=', 1)[1] for part in hashes.split(',') if part.strip().startswith('md5='))
        if not archive.exists() or digest(archive) != expected:
            partial = archive.with_suffix('.zip.part')
            with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            if digest(partial) != expected:
                raise RuntimeError(f'{model}: checksum mismatch; partial file retained for diagnosis')
            partial.replace(archive)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                destination = (ROOT / member.filename).resolve()
                if not destination.is_relative_to(ROOT / model):
                    raise ValueError(f'Unexpected archive path: {member.filename}')
            zipped.extractall(ROOT)
        print(f'{model}: verified and extracted', flush=True)


if __name__ == '__main__':
    main()
