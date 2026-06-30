import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


def _find_bin(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    for d in ("/opt/homebrew/opt/postgresql@16/bin", "/opt/homebrew/lib/postgresql/bin", "/usr/lib/postgresql/16/bin"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """Spin up a throwaway Postgres cluster in a temp dir for the session."""
    pg_ctl = _find_bin("pg_ctl")
    initdb = _find_bin("initdb")
    if not pg_ctl or not initdb:
        pytest.skip("postgres binaries not installed — install postgresql to run these tests")
    pgdata = Path(tempfile.mkdtemp(prefix="pgvector-test-"))
    subprocess.run([initdb, "-D", str(pgdata), "--auth=trust", "--no-locale", "--encoding=UTF8"], check=True, capture_output=True)
    # Create databases we need
    port = "55432"
    proc = subprocess.Popen(
        [pg_ctl, "-D", str(pgdata), "-o", f"-p {port} -k /tmp", "start"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    proc.wait(timeout=60)
    # Wait for ready
    psql = _find_bin("psql")
    for _ in range(30):
        r = subprocess.run([psql, "-h", "/tmp", "-p", port, "-d", "postgres", "-c", "SELECT 1"],
                           capture_output=True)
        if r.returncode == 0:
            break
        time.sleep(0.5)
    subprocess.run([psql, "-h", "/tmp", "-p", port, "-d", "postgres", "-c", "CREATE DATABASE test_pgvector"], check=True)
    subprocess.run([psql, "-h", "/tmp", "-p", port, "-d", "test_pgvector", "-c", "CREATE EXTENSION vector"], check=True)
    dsn = f"postgresql://localhost:{port}/test_pgvector"
    os.environ["KIWIX_TEST_DSN"] = dsn
    yield dsn
    subprocess.run([pg_ctl, "-D", str(pgdata), "stop", "-m", "fast"], capture_output=True)
    shutil.rmtree(pgdata, ignore_errors=True)
