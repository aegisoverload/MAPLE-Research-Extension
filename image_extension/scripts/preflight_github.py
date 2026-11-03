"""Best-effort GitHub upload preflight. Not a substitute for secret scanning or license review."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_SEGMENTS = {'.venv', 'data', 'private_data', 'donated_data', 'datasets', 'runs', 'results', 'models', 'checkpoints', 'outputs', 'secrets'}
SENSITIVE_NAMES = {'.env', '.env.local', 'credentials.json'}
SECRET_PATTERNS = {
    'Google API key pattern': re.compile(r'AIza[0-9A-Za-z_-]{30,}'),
    'GitHub PAT pattern': re.compile(r'gh[pousr]_[a-zA-Z0-9_]{20,}'),
    'OpenAI API key pattern': re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{24,}'),
}


def files_for_scan():
    try:
        r = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'],
                           check=True, capture_output=True)
        return [ROOT / x.decode('utf-8', errors='ignore') for x in r.stdout.split(b'\0') if x]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [p for p in ROOT.rglob('*') if p.is_file() and not any(s in SENSITIVE_SEGMENTS for s in p.relative_to(ROOT).parts)]


def main():
    issues = []
    for path in files_for_scan():
        rel = path.relative_to(ROOT)
        if any(part in SENSITIVE_SEGMENTS for part in rel.parts) or path.name in SENSITIVE_NAMES:
            issues.append(f'Sensitive/ignored path IS TRACKED: {rel}')
            continue
        if path.stat().st_size > 2_000_000:
            issues.append(f'Unusually large tracked file: {rel}')
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except (UnicodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                issues.append(f'Possible {label} in {rel}')
    if issues:
        print('UPLOAD PRE-FLIGHT FAILED:')
        for issue in issues:
            print(' -', issue)
        raise SystemExit(1)
    print('PRE-FLIGHT PASSED (best-effort only). Check licenses, Git history, and secrets manually.')


if __name__ == '__main__':
    main()
