"""
Tests for web_ui authentication: bcrypt hashing, legacy migration, fail-closed
seeding, and the password policy. These exercise AuthManager directly and do not
require Streamlit (auth.py imports it defensively).
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# auth.py lives in web_ui/ which is not an installed package; load it by path.
_AUTH_PATH = Path(__file__).resolve().parent.parent / "web_ui" / "auth.py"
_spec = importlib.util.spec_from_file_location("deep_eagle_auth", _AUTH_PATH)
auth = importlib.util.module_from_spec(_spec)
sys.modules["deep_eagle_auth"] = auth
_spec.loader.exec_module(auth)


def _users_file(tmp_path):
    return str(tmp_path / "users.json")


def test_bcrypt_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEP_EAGLE_ADMIN_PASSWORD", raising=False)
    mgr = auth.AuthManager(users_file=_users_file(tmp_path))
    assert mgr.add_user("alice", "correcthorsebattery") is True

    # Stored as bcrypt (not plaintext, not sha256)
    stored = mgr.users["alice"]
    assert stored.startswith("$2")
    assert auth.AuthManager._is_bcrypt_hash(stored)

    assert mgr.verify_credentials("alice", "correcthorsebattery") is True
    assert mgr.verify_credentials("alice", "wrong-password") is False
    assert mgr.verify_credentials("nobody", "whatever") is False


def test_legacy_sha256_is_upgraded(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEP_EAGLE_ADMIN_PASSWORD", raising=False)
    path = _users_file(tmp_path)
    # Seed a legacy SHA-256 (fixed-salt) hash, the old scheme
    legacy = hashlib.sha256(b"deep_eagle_salt_v1longenoughpassword").hexdigest()
    Path(path).write_text(json.dumps({"bob": legacy}))

    mgr = auth.AuthManager(users_file=path)
    assert not auth.AuthManager._is_bcrypt_hash(mgr.users["bob"])

    # Logging in with the right password succeeds AND upgrades the hash
    assert mgr.verify_credentials("bob", "longenoughpassword") is True
    assert mgr.users["bob"].startswith("$2")
    # Persisted upgrade
    assert json.loads(Path(path).read_text())["bob"].startswith("$2")
    # Subsequent login still works against the bcrypt hash
    assert mgr.verify_credentials("bob", "longenoughpassword") is True


def test_fail_closed_when_no_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEP_EAGLE_ADMIN_PASSWORD", raising=False)
    mgr = auth.AuthManager(users_file=_users_file(tmp_path))
    # No users file and no configured secret -> no users (fail closed)
    assert mgr.users == {}
    assert mgr.verify_credentials("admin", "admin123") is False


def test_admin_seeded_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEP_EAGLE_ADMIN_PASSWORD", "a-strong-admin-password")
    mgr = auth.AuthManager(users_file=_users_file(tmp_path))
    assert "admin" in mgr.users
    assert mgr.users["admin"].startswith("$2")
    assert mgr.verify_credentials("admin", "a-strong-admin-password") is True


def test_password_policy_enforced(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEP_EAGLE_ADMIN_PASSWORD", raising=False)
    mgr = auth.AuthManager(users_file=_users_file(tmp_path))
    with pytest.raises(ValueError, match="at least"):
        mgr.add_user("shorty", "short")

    mgr.add_user("carol", "initial-strong-password")
    with pytest.raises(ValueError, match="at least"):
        mgr.change_password("carol", "initial-strong-password", "weak")
