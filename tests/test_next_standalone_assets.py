import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def test_frontend_build_syncs_static_and_public_assets(tmp_path: Path) -> None:
    package = json.loads((FRONTEND / "package.json").read_text())
    assert package["scripts"]["postbuild"] == "node scripts/sync-standalone-assets.mjs"

    next_dir = tmp_path / ".next"
    public_dir = tmp_path / "public"
    (next_dir / "static/css").mkdir(parents=True)
    (next_dir / "standalone/.next/static").mkdir(parents=True)
    (next_dir / "standalone/public").mkdir()
    (public_dir / "icons").mkdir(parents=True)
    (next_dir / "static/css/app.css").write_text("body { color: red; }")
    (public_dir / "manifest.json").write_text('{"name":"test"}')
    (public_dir / "icons/icon.png").write_bytes(b"png")
    (next_dir / "standalone/.next/static/stale.js").write_text("stale")
    (next_dir / "standalone/public/stale.txt").write_text("stale")
    static_inode = (next_dir / "standalone/.next/static").stat().st_ino
    public_inode = (next_dir / "standalone/public").stat().st_ino

    subprocess.run(
        [
            "node",
            str(FRONTEND / "scripts/sync-standalone-assets.mjs"),
            str(next_dir),
            str(public_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (next_dir / "standalone/.next/static/css/app.css").read_text() == "body { color: red; }"
    assert (next_dir / "standalone/public/manifest.json").read_text() == '{"name":"test"}'
    assert (next_dir / "standalone/public/icons/icon.png").read_bytes() == b"png"
    assert not (next_dir / "standalone/.next/static/stale.js").exists()
    assert not (next_dir / "standalone/public/stale.txt").exists()
    assert (next_dir / "standalone/.next/static").stat().st_ino == static_inode
    assert (next_dir / "standalone/public").stat().st_ino == public_inode


def test_docker_build_can_skip_postbuild_asset_sync(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "node",
            str(FRONTEND / "scripts/sync-standalone-assets.mjs"),
            str(tmp_path / ".next"),
            str(tmp_path / "public"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "SKIP_STANDALONE_ASSET_SYNC": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert "Skipped standalone asset sync" in result.stdout
    assert "SKIP_STANDALONE_ASSET_SYNC=1 npm run build" in (ROOT / "Dockerfile").read_text()
    assert "SKIP_STANDALONE_ASSET_SYNC=1 npm run build" in (
        FRONTEND / "Dockerfile.web"
    ).read_text()
