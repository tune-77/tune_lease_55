#!/usr/bin/env python3
"""Cloud Run startup: SSH key setup, git clone (demo.db/mind.json), GCS ChromaDB sync."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA_GIT_DIR = os.environ.get("DATA_GIT_DIR", "/app/data-git")
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
CHROMA_LOCAL = "/app/api/chroma_db"


def _get_project_id() -> str:
    proj = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT", "")
    if proj:
        return proj
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.read().decode()
    except Exception:
        return ""


def setup_ssh_key() -> bool:
    """Secret Manager から SSH 秘密鍵を取得して ~/.ssh/id_ed25519 に書き込む。"""
    project_id = _get_project_id()
    if not project_id:
        print("[cloud_init] GCP project ID 不明、SSH key セットアップをスキップ")
        return False
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/github-deploy-key/versions/1"
        response = client.access_secret_version(name=name)
        key_data = response.payload.data

        ssh_dir = Path.home() / ".ssh"
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
        key_path = ssh_dir / "id_ed25519"
        key_path.write_bytes(key_data)
        key_path.chmod(0o600)

        known_hosts = ssh_dir / "known_hosts"
        keyscan = subprocess.run(
            ["ssh-keyscan", "-H", "github.com"],
            capture_output=True, timeout=15,
        )
        if keyscan.returncode == 0:
            with open(known_hosts, "a") as f:
                f.write(keyscan.stdout.decode())

        print("[cloud_init] SSH key セットアップ完了")
        return True
    except ImportError:
        print("[cloud_init] google-cloud-secret-manager が未インストール、スキップ")
        return False
    except Exception as e:
        print(f"[cloud_init] SSH key セットアップ失敗: {e}")
        return False


def clone_or_pull_repo() -> bool:
    """デモDB・mind.json を含む git リポジトリを clone / pull する。"""
    git_dir = Path(DATA_GIT_DIR)
    try:
        if (git_dir / ".git").exists():
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(git_dir), capture_output=True, timeout=60,
            )
            print(f"[cloud_init] git pull: returncode={result.returncode}")
        else:
            git_dir.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--depth=1", GITHUB_REPO, str(git_dir)],
                capture_output=True, timeout=120,
            )
            print(f"[cloud_init] git clone: returncode={result.returncode}")
        if result.returncode != 0:
            print(f"[cloud_init] git error: {result.stderr.decode(errors='replace')}")
            return False
        return True
    except Exception as e:
        print(f"[cloud_init] git 操作失敗: {e}")
        return False


def sync_data_files() -> None:
    """git clone から mind.json を DATA_DIR にコピーする。"""
    src_dir = Path(DATA_GIT_DIR) / "data"
    dst_dir = Path(DATA_DIR)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("mind.json",):
        src = src_dir / filename
        dst = dst_dir / filename
        if src.exists():
            shutil.copy2(str(src), str(dst))
            print(f"[cloud_init] {filename} → {dst}")


def sync_chromadb() -> None:
    """GCS から ChromaDB をローカルに同期する。"""
    if not GCS_BUCKET:
        print("[cloud_init] GCS_BUCKET 未設定、ChromaDB 同期をスキップ")
        return
    local_dir = Path(CHROMA_LOCAL)
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from google.cloud import storage
        bucket_name = GCS_BUCKET.replace("gs://", "").split("/")[0]
        prefix = "chromadb/"
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            rel = blob.name[len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
        print(f"[cloud_init] ChromaDB 同期完了 ({len(blobs)} オブジェクト) → {local_dir}")
    except ImportError:
        print("[cloud_init] google-cloud-storage が未インストール、ChromaDB 同期をスキップ")
    except Exception as e:
        print(f"[cloud_init] ChromaDB 同期失敗（非致命的）: {e}")


def main() -> None:
    if GITHUB_REPO:
        ssh_ok = setup_ssh_key()
        if ssh_ok and clone_or_pull_repo():
            sync_data_files()
        else:
            print("[cloud_init] git 同期をスキップ")
    else:
        print("[cloud_init] GITHUB_REPO 未設定、git 同期をスキップ")

    sync_chromadb()
    print("[cloud_init] 初期化完了")


if __name__ == "__main__":
    main()
