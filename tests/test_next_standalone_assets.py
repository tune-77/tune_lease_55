import json
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
    (next_dir / "standalone").mkdir()
    (public_dir / "icons").mkdir(parents=True)
    (next_dir / "static/css/app.css").write_text("body { color: red; }")
    (public_dir / "manifest.json").write_text('{"name":"test"}')
    (public_dir / "icons/icon.png").write_bytes(b"png")

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
