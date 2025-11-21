# Quick Start - Deploy Deep-Eagle Dashboard

## 🚀 Deploy in 5 Minutes

### Step 1: Push to GitHub (if not already done)
```bash
git add .
git commit -m "Add authentication and deployment config"
git push origin main
```

### Step 2: Go to Streamlit Cloud
1. Visit: https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"

### Step 3: Configure Your App
- **Repository**: `mckayje3/deep-eagle`
- **Branch**: `main`
- **Main file path**: `web_ui/app.py`

### Step 4: Deploy
Click "Deploy" and wait 2-3 minutes

### Step 5: Login
**Default Credentials:**
- Username: `admin`
- Password: `admin123`

### Step 6: Change Password (IMPORTANT!)
1. Go to Settings page
2. Expand "User Management"
3. Change password immediately

## 🎉 Done!

Your dashboard is now accessible from anywhere!

Share the URL with your team and create accounts for them in Settings → User Management.

---

## 📱 Your Dashboard URL

After deployment, your URL will be:
```
https://[your-username]-deep-eagle-web-ui-app-[random].streamlit.app
```

Bookmark it for easy access!

---

## ➕ Add Users

As admin, you can add team members:
1. Login as admin
2. Go to Settings
3. Expand "User Management"
4. Expand "Add New User"
5. Enter username and password
6. Click "Add User"

---

## 🔧 Use in Other Apps

To use deep-eagle library in your other Streamlit apps:

**In `requirements.txt`:**
```
git+https://github.com/mckayje3/deep-eagle.git
```

**In your code:**
```python
from core import LSTMModel, TimeSeriesDataset, Trainer

# Use as normal
model = LSTMModel(input_dim=10, hidden_dim=64, output_dim=1)
```

That's it!
