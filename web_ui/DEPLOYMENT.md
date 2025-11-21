# Deploying Deep-Eagle Dashboard to Streamlit Cloud

This guide walks you through deploying the Deep-Eagle dashboard to Streamlit Cloud with authentication.

## Prerequisites

1. GitHub account with the `deep-eagle` repository
2. Streamlit Cloud account (free at https://share.streamlit.io)

## Step 1: Prepare Your Repository

Ensure all files are committed to GitHub:

```bash
cd web_ui
git add .
git commit -m "Add authentication and deployment config"
git push origin main
```

## Step 2: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your repository: `mckayje3/deep-eagle`
4. Set the following:
   - **Main file path**: `web_ui/app.py`
   - **Branch**: `main` (or your preferred branch)
   - **Python version**: 3.9 or higher

5. Click "Deploy!"

## Step 3: Configure Secrets (Optional)

If you want to customize passwords via Streamlit Cloud secrets:

1. In your Streamlit Cloud dashboard, go to your app
2. Click "Settings" → "Secrets"
3. Add secrets in TOML format:

```toml
[auth]
admin_password = "your-secure-password-here"
```

## Step 4: Access Your Dashboard

Once deployed, you'll get a URL like:
```
https://mckayje3-deep-eagle-web-ui-app-xxxxxx.streamlit.app
```

## Default Credentials

- **Username**: `admin`
- **Password**: `admin123`

**IMPORTANT**: Change your password immediately after first login!
1. Log in with default credentials
2. Go to "Settings" page
3. Expand "User Management"
4. Use "Change Password" to set a secure password

## Managing Users

As an admin, you can:
- **Change your password**: Settings → User Management → Change Password
- **Add new users**: Settings → User Management → Add New User
- **Delete users**: Settings → User Management → Manage Users

## Security Notes

1. **Change default password immediately** after deployment
2. User credentials are stored in `.streamlit/users.json` (persistent on Streamlit Cloud)
3. Passwords are hashed using SHA-256
4. **Do not commit** `.streamlit/secrets.toml` or `.streamlit/users.json` to git

## Troubleshooting

### Authentication not working
- Clear browser cookies and cache
- Try incognito/private browsing mode
- Check that `auth.py` was deployed correctly

### Import errors
- Verify `requirements.txt` includes all dependencies
- Check that parent directory path is added correctly in `app.py`

### Can't access certain pages
- Ensure you're logged in
- Check session state hasn't expired
- Refresh the page

## Updating the App

To update your deployed app:

1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Your update message"
   git push origin main
   ```
3. Streamlit Cloud will automatically redeploy (takes 2-3 minutes)

## Custom Domain (Optional)

To use a custom domain:

1. In Streamlit Cloud, go to Settings → General
2. Add your custom domain
3. Configure DNS records as instructed
4. Enable HTTPS

## Monitoring

Streamlit Cloud provides:
- **Logs**: View application logs in the dashboard
- **Analytics**: Basic usage statistics
- **Health checks**: Automatic monitoring and restarts

## Support

- Streamlit Documentation: https://docs.streamlit.io
- Deep-Eagle Issues: https://github.com/mckayje3/deep-eagle/issues
