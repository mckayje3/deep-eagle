"""
Authentication module for Deep-TimeSeries Dashboard
Handles user login and session management.

Security notes:
- Passwords are hashed with bcrypt (per-user salt, tunable work factor).
- Legacy SHA-256 hashes are transparently upgraded to bcrypt on next login.
- There is NO baked-in default password. The initial admin password must be
  supplied out-of-band via the DEEP_EAGLE_ADMIN_PASSWORD environment variable or
  Streamlit secrets ([auth] admin_password). If neither is set and no users file
  exists, authentication fails closed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path

import bcrypt

try:  # Streamlit is only present in the web UI; keep AuthManager usable headless.
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised only outside the app
    st = None

logger = logging.getLogger(__name__)

# Minimum password length enforced for user-set passwords.
MIN_PASSWORD_LENGTH = 12
# Idle/session lifetime before re-authentication is required.
SESSION_TTL_SECONDS = 8 * 60 * 60

_LEGACY_SALT = "deep_eagle_salt_v1"  # only used to verify+upgrade old hashes


def _get_configured_admin_password() -> str | None:
    """Return the out-of-band initial admin password, if configured."""
    env = os.environ.get("DEEP_EAGLE_ADMIN_PASSWORD")
    if env:
        return env
    if st is not None:
        try:
            return st.secrets["auth"]["admin_password"]
        except Exception:  # secrets file may be absent or key missing
            return None
    return None


class AuthManager:
    """Manages user authentication and sessions."""

    def __init__(self, users_file: str = ".streamlit/users.json"):
        """
        Args:
            users_file: Path to the users configuration file.
        """
        self.users_file = Path(users_file)
        self.users = self._load_users()

    def _load_users(self) -> dict[str, str]:
        """Load users from disk, seeding an admin only if one is configured."""
        if self.users_file.exists():
            try:
                with open(self.users_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load users file: {e}")
                return {}

        # No users file: seed an admin ONLY from an out-of-band secret.
        admin_password = _get_configured_admin_password()
        if admin_password:
            users = {"admin": self._hash_password(admin_password)}
            self.users = users
            self._save_users()
            logger.info("Seeded initial 'admin' user from configured secret.")
            return users

        # Fail closed: no users and nothing configured.
        logger.warning(
            "No users file and no DEEP_EAGLE_ADMIN_PASSWORD / [auth].admin_password "
            "configured — authentication is not available."
        )
        return {}

    def _save_users(self) -> None:
        """Persist users to disk."""
        try:
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.users_file, "w") as f:
                json.dump(self.users, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save users: {e}")
            if st is not None:
                st.error(f"Failed to save users: {e}")

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash a password with bcrypt (per-user salt embedded in the result)."""
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _is_bcrypt_hash(stored: str) -> bool:
        return stored.startswith(("$2a$", "$2b$", "$2y$"))

    @staticmethod
    def _legacy_sha256(password: str) -> str:
        return hashlib.sha256(f"{_LEGACY_SALT}{password}".encode()).hexdigest()

    def verify_credentials(self, username: str, password: str) -> bool:
        """
        Verify credentials. Legacy SHA-256 hashes are upgraded to bcrypt on a
        successful match.

        Returns:
            True if the credentials are valid.
        """
        stored = self.users.get(username)
        if stored is None:
            return False

        if self._is_bcrypt_hash(stored):
            try:
                return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
            except ValueError:
                return False

        # Legacy SHA-256 path: verify with the old scheme, then upgrade.
        if secrets.compare_digest(stored, self._legacy_sha256(password)):
            self.users[username] = self._hash_password(password)
            self._save_users()
            logger.info(f"Upgraded password hash for user '{username}' to bcrypt.")
            return True
        return False

    @staticmethod
    def check_password_policy(password: str) -> None:
        """Raise ValueError if the password does not meet the policy."""
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

    def add_user(self, username: str, password: str) -> bool:
        """
        Add a new user. Enforces the password policy.

        Returns:
            True if added, False if the username already exists.

        Raises:
            ValueError: If the password violates the policy.
        """
        if username in self.users:
            return False
        self.check_password_policy(password)
        self.users[username] = self._hash_password(password)
        self._save_users()
        return True

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Change a user's password. Enforces the password policy on the new password.

        Returns:
            True if changed, False if the current password is incorrect.

        Raises:
            ValueError: If the new password violates the policy.
        """
        if not self.verify_credentials(username, old_password):
            return False
        self.check_password_policy(new_password)
        self.users[username] = self._hash_password(new_password)
        self._save_users()
        return True

    def delete_user(self, username: str) -> bool:
        """Delete a user. Returns True if a user was removed."""
        if username not in self.users:
            return False
        del self.users[username]
        self._save_users()
        return True


def require_authentication() -> str | None:
    """
    Ensure the user is authenticated, showing a login form if not.

    Returns:
        The username if authenticated, otherwise None.
    """
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.login_time = 0.0

    # Expire stale sessions
    if st.session_state.authenticated:
        age = time.time() - st.session_state.get("login_time", 0.0)
        if age > SESSION_TTL_SECONDS:
            st.session_state.authenticated = False
            st.session_state.username = None
            st.warning("Your session has expired. Please log in again.")
        else:
            return st.session_state.username

    # Show login form
    st.title("🔐 Login Required")
    st.markdown("Please log in to access the Deep-TimeSeries Dashboard")

    auth_manager = AuthManager()

    if not auth_manager.users:
        st.error(
            "Authentication is not configured. Set an initial admin password via the "
            "`DEEP_EAGLE_ADMIN_PASSWORD` environment variable or the `[auth] admin_password` "
            "Streamlit secret, then reload."
        )
        return None

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if auth_manager.verify_credentials(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.login_time = time.time()
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    return None


def logout():
    """Log out the current user."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.login_time = 0.0
    st.rerun()


def show_user_management():
    """Show the user management interface (admin only)."""
    if not st.session_state.get("authenticated", False):
        st.warning("You must be logged in to manage users")
        return

    auth_manager = AuthManager()

    st.subheader("User Management")

    # Change password
    with st.expander("Change Password"):
        with st.form("change_password_form"):
            old_password = st.text_input("Current Password", type="password", key="old_pw")
            new_password = st.text_input("New Password", type="password", key="new_pw")
            confirm_password = st.text_input(
                "Confirm New Password", type="password", key="confirm_pw"
            )

            if st.form_submit_button("Change Password"):
                if new_password != confirm_password:
                    st.error("New passwords do not match")
                else:
                    try:
                        if auth_manager.change_password(
                            st.session_state.username, old_password, new_password
                        ):
                            st.success("Password changed successfully!")
                        else:
                            st.error("Current password is incorrect")
                    except ValueError as e:
                        st.error(str(e))

    # Add new user (admin only)
    if st.session_state.username == "admin":
        with st.expander("Add New User"):
            with st.form("add_user_form"):
                new_username = st.text_input("Username", key="new_username")
                new_user_password = st.text_input("Password", type="password", key="new_user_pw")

                if st.form_submit_button("Add User"):
                    if len(new_username) < 3:
                        st.error("Username must be at least 3 characters")
                    else:
                        try:
                            if auth_manager.add_user(new_username, new_user_password):
                                st.success(f"User '{new_username}' added successfully!")
                            else:
                                st.error(f"Username '{new_username}' already exists")
                        except ValueError as e:
                            st.error(str(e))

        # List and delete users
        with st.expander("Manage Users"):
            st.write("**Registered Users:**")
            for username in auth_manager.users.keys():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(username)
                with col2:
                    if username != "admin" and username != st.session_state.username:
                        if st.button("Delete", key=f"del_{username}"):
                            if auth_manager.delete_user(username):
                                st.success(f"User '{username}' deleted")
                                st.rerun()
