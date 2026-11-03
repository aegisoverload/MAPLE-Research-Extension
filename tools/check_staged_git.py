"""Best-effort pre-push stage scan. Not an audit of all Git history or permissions."""
from pathlib import Path
import re, subprocess, sys

BLOCKED_PARTS = {'.venv','venv','secrets','results','runs','checkpoints','models'}
BAD_NAMES = {'.env','.env.local','credentials.json'}
PATTERNS = [
    re.compile(r'AIza[0-9A-Za-z_-]{30,}'),
    re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),
    re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{24,}'),
]

def main():
    proc = subprocess.run(['git','diff','--cached','--name-only','--diff-filter=ACM','-z'],capture_output=True,check=True)
    names = [n.decode('utf-8','replace') for n in proc.stdout.split(b'\0') if n]
    if not names:
        print('Nothing staged. Run git add and review git status first.');return
    bad=[]
    for name in names:
        p=Path(name); parts=set(p.parts)
        if p.name in BAD_NAMES or parts & BLOCKED_PARTS:
            bad.append(f'BLOCKED PATH: {name}');continue
        if not p.exists() or not p.is_file():continue
        if p.stat().st_size > 50_000_000:bad.append(f'LARGE FILE (>50 MB): {name}')
        if p.stat().st_size < 5_000_000:
            try: s=p.read_text(encoding='utf-8')
            except (OSError,UnicodeError):continue
            if any(pattern.search(s) for pattern in PATTERNS):bad.append(f'POSSIBLE API SECRET: {name}')
    if bad:
        print('PREFLIGHT FAILED:\n'+'\n'.join(bad));sys.exit(1)
    print(f'PREFLIGHT PASSED on {len(names)} staged paths (best-effort).')
    print('REMINDER: inspect full Git history and licenses separately; private/restricted data may be tracked upstream.')

if __name__=='__main__':main()
