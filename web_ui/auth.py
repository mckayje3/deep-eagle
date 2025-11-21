"""
Authentication module for Deep-TimeSeries Dashboard
Handles user login and session management
"""

import streamlit as st
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict


class AuthManager:
    """Manages user authentication and sessions"""

    def __init__(self, users_file: str = ".streamlit/users.json"):
        """
        Initialize authentication manager

        Args:
            users_file: Path to users configuration file
        """
        self.users_file = Path(users_file)
        self.users = self._load_users()

    def _load_users(self) -> Dict[str, str]:
        """Load users from configuration file"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

        # Default users if file doesn't exist
        # Format: {"username": "hashed_password"}
        return {
            "admin": self._hash_password("admin123"),
        }

    def _save_users(self):
        """Save users to configuration file"""
        try:
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save users: {e}")

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_credentials(self, username: str, password: str) -> bool:
        """
        Verify user credentials

        Args:
            username: Username to verify
            password: Password to verify

        Returns:
            True if credentials are valid, False otherwise
        """
        if username not in self.users:
            return False

        hashed_password = self._hash_password(password)
        return self.users[username] == hashed_password

    def add_user(self, username: str, password: str) -> bool:
        """
        Add a new user

        Args:
            username: Username for new user
            password: Password for new user

        Returns:
            True if user was added, False if username exists
        """
        if username in self.users:
            return False

        self.users[username] = self._hash_password(password)
        self._save_users()
        return True

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Change user password

        Args:
            username: Username to change password for
            old_password: Current password
            new_password: New password

        Returns:
            True if password was changed, False otherwise
        """
        if not self.verify_credentials(username, old_password):
            return False

        self.users[username] = self._hash_password(new_password)
        self._save_users()
        return True

    def delete_user(self, username: str) -> bool:
        """
        Delete a user

        Args:
            username: Username to delete

        Returns:
            True if user was deleted, False if user doesn't exist
        """
        if username not in self.users:
            return False

        del self.users[username]
        self._save_users()
        return True


def require_authentication() -> Optional[str]:
    """
    Check if user is authenticated, show login form if not

    Returns:
        Username if authenticated, None otherwise
    """
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    # If already authenticated, return username
    if st.session_state.authenticated:
        return st.session_state.username

    # Show login form
    st.title("🔐 Login Required")
    st.markdown("Please log in to access the Deep-TimeSeries Dashboard")

    auth_manager = AuthManager()

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if auth_manager.verify_credentials(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.markdown("---")
    st.info("**Default credentials:**\n- Username: `admin`\n- Password: `admin123`\n\n"
            "Change your password after first login in the Settings page.")

    return None


def logout():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()


def show_user_management():
    """Show user management interface (admin only)"""
    if not st.session_state.get('authenticated', False):
        st.warning("You must be logged in to manage users")
        return

    auth_manager = AuthManager()

    st.subheader("User Management")

    # Change password
    with st.expander("Change Password"):
        with st.form("change_password_form"):
            old_password = st.text_input("Current Password", type="password", key="old_pw")
            new_password = st.text_input("New Password", type="password", key="new_pw")
            confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pw")

            if st.form_submit_button("Change Password"):
                if new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif auth_manager.change_password(st.session_state.username, old_password, new_password):
                    st.success("Password changed successfully!")
                else:
                    st.error("Current password is incorrect")

    # Add new user (admin only)
    if st.session_state.username == "admin":
        with st.expander("Add New User"):
            with st.form("add_user_form"):
                new_username = st.text_input("Username", key="new_username")
                new_user_password = st.text_input("Password", type="password", key="new_user_pw")

                if st.form_submit_button("Add User"):
                    if len(new_username) < 3:
                        st.error("Username must be at least 3 characters")
                    elif len(new_user_password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif auth_manager.add_user(new_username, new_user_password):
                        st.success(f"User '{new_username}' added successfully!")
                    else:
                        st.error(f"Username '{new_username}' already exists")

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
