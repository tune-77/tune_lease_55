"""Run the installer with isolated paths and mocked macOS service commands."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def installer(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "install_next_launchagent.sh"
    # Redirect only the installer's destination, leaving the user's HOME untouched.
    source = (root / "scripts/install_next_launchagent.sh").read_text()
    script.write_text(source.replace("$HOME/Library/LaunchAgents", "$TEST_LAUNCHAGENTS_DIR"))
    launchd = scripts / "launchd"
    launchd.mkdir()
    (launchd / "com.tunelease.next.plist").write_text("test plist")
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("plutil", "launchctl"):
        command = commands / name
        command.write_text('#!/bin/sh\nprintf "%s\\n" "called" >> "$TEST_PLATFORM_LOG"\n')
        command.chmod(0o755)
    config = tmp_path / "config"
    config.mkdir()
    credential = config / "public_tunnel_auth"
    credential.write_text("existing-password")
    log = tmp_path / "platform.log"

    def run(password, *, interactive=False):
        env = {**os.environ,
            "PATH": f"{commands}:{os.environ['PATH']}",
            "TUNELEASE_CONFIG_DIR": str(config),
            "TEST_LAUNCHAGENTS_DIR": str(tmp_path / "agents"),
            "TEST_PLATFORM_LOG": str(log), "PUBLIC_TUNNEL_AUTH": password,
            "LC_ALL": "C.UTF-8",
        }
        if interactive:
            import pty
            import termios
            credential.unlink()
            env.pop("PUBLIC_TUNNEL_AUTH")
            master, slave = pty.openpty()
            settings = termios.tcgetattr(slave)
            settings[3] &= ~termios.ECHO
            termios.tcsetattr(slave, termios.TCSANOW, settings)
            try:
                process = subprocess.Popen(["bash", str(script)], env=env,
                    stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                os.write(master, (password + "\n").encode("utf-8"))
                try:
                    stdout, stderr = process.communicate(timeout=10)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
                return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
            finally:
                os.close(master)
                os.close(slave)
        return subprocess.run(["bash", str(script)], env=env,
            capture_output=True, text=True, timeout=10)
    return run, credential, log


@pytest.mark.parametrize("password", [
    "あいうえおかきくけこさし", "abcdefghijkl😀", "abcdefghijklé",
    "abcdef\tghijkl", "abcdefghijkl\x7f", "abcdefghijkl\x01",
    "abcdefghijkl\n", "abcdefghijkl\r", "short-pass",
])
def test_invalid_password_is_not_persisted_or_started(installer, password):
    run, credential, log = installer
    result = run(password)
    assert result.returncode != 0
    assert credential.read_text() == "existing-password"
    assert not log.exists()
    assert password not in result.stdout + result.stderr
    assert not list(credential.parent.glob(".public_tunnel_auth.*"))


@pytest.mark.parametrize("password", [
    "abcdefghijkl", "ABC123!@#$%^&*()_+-=[]{}:;,.?/~", ' spaces " \\ end ',
])
def test_printable_ascii_password_is_saved_exactly(installer, password):
    run, credential, log = installer
    result = run(password)
    assert result.returncode == 0, result.stderr
    assert credential.read_text() == password
    assert credential.stat().st_mode & 0o777 == 0o600
    assert log.exists()
    assert password not in result.stdout + result.stderr


@pytest.mark.parametrize("password,accepted", [("あいうえおかきくけこさし", False), ("abcdefghijkl", True)])
def test_interactive_installer_validates_before_provisioning(installer, password, accepted):
    run, credential, log = installer
    result = run(password, interactive=True)
    assert (result.returncode == 0) == accepted
    assert credential.exists() == accepted
    assert log.exists() == accepted
    assert password not in result.stdout + result.stderr
