"""Verify a standalone paper handoff without accessing the source workspace."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def safe_file(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('unsafe package path: ' + name)
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('missing or external package file: ' + name)
    return path


def verify(root):
    root = Path(root).resolve(strict=True)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('symlinks are not allowed in the standalone package')
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if manifest['schema'] != 'edgefm.portable_paper_handoff/1':
        raise ValueError('invalid package schema')
    expected = set()
    for record in manifest['files']:
        name = record['path']
        if name in expected:
            raise ValueError('duplicate package entry: ' + name)
        expected.add(name)
        path = safe_file(root, name)
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('package content changed: ' + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != expected | {'MANIFEST.json', 'SHA256SUMS'}:
        raise ValueError('unlisted or missing package files')
    sums = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        if name in sums or len(sha) != 64:
            raise ValueError('invalid checksum entry')
        sums[name] = sha
    if set(sums) != expected | {'MANIFEST.json'}:
        raise ValueError('checksum file set differs')
    for name, sha in sums.items():
        if digest(safe_file(root, name)) != sha:
            raise ValueError('checksum mismatch: ' + name)
    links = 0
    for name in sorted(expected):
        if not name.endswith('.md'):
            continue
        path = root / name
        for match in re.finditer(r'\[[^\]]*\]\(([^)\n]+)\)', path.read_text()):
            url = urlsplit(match.group(1).strip().strip('<>'))
            if url.scheme or not url.path:
                continue
            target = Path(unquote(url.path))
            if target.is_absolute():
                raise ValueError('absolute local Markdown link in ' + name)
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(root) or not resolved.exists():
                raise ValueError('broken or escaping Markdown link in ' + name + ': ' + url.path)
            links += 1
    forbidden = {'.hbm', '.engine', '.onnx', '.safetensors', '.pt', '.pth', '.so', '.bin'}
    if any(Path(name).suffix in forbidden for name in expected):
        raise ValueError('model/build payload unexpectedly included')
    if manifest.get('self_contained_without_machine_references'):
        markers = re.compile(r'(?<![\w])/(?:home|xs-train-nas|data/datalog|tmp|root|usr|app|opt|var|run|sys|proc|mnt)/|\b(?:https?|ssh|sftp|file)://|\b(?:10\.\d+\.\d+\.\d+|127\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)')
        for name in [*expected, 'MANIFEST.json']:
            if Path(name).suffix in ('.md', '.json', '.csv') and markers.search((root / name).read_text(encoding='utf-8-sig')):
                raise ValueError('machine path or connection reference remains: ' + name)
    return dict(status='portable-paper-package-verified', files=len(expected), local_markdown_links=links,
        bytes=sum(row['bytes'] for row in manifest['files']), original_workspace_required=False,
        model_inference=False, numerical_experiment_rerun=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False))
