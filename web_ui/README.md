# Deep-Eagle Dashboard 🦅

A comprehensive web-based dashboard for the Deep-TimeSeries framework, providing a **secure**, intuitive visual interface for dataset management, model building, training, and evaluation.

## 🔐 Security Features

- **User Authentication**: Secure login system with password hashing
- **Session Management**: Persistent user sessions
- **Multi-user Support**: Admin can create and manage multiple users
- **Password Management**: Users can change passwords securely
- **Access Control**: Only authenticated users can access the dashboard

## Features

### 🏠 Home Dashboard
- Quick start guide
- Usage analytics overview
- Framework status and metrics
- Recent activity

### 📊 Dataset Manager
- Upload CSV/Excel files
- Load example datasets
- Data preview and statistics
- Interactive visualizations
- Missing value analysis and handling
- Data export capabilities

### 🏗️ Model Builder
- Visual model configuration
- Support for LSTM, GRU, and Transformer architectures
- Quick presets (Fast Prototype, Balanced, High Capacity)
- Hyperparameter tuning
- Parameter estimation
- Configuration save/load

### 🚀 Training
- Real-time training monitoring
- Live loss curves
- Progress tracking
- Model checkpointing
- Early stopping configuration
- Training history export

### 📈 Results & Evaluation
- Training curve visualization
- Performance metrics (MSE, RMSE, MAE, MAPE)
- Predictions vs. actual plots
- Error analysis
- Statistical summaries
- Results export

### 🔮 Prediction
- Multiple input methods (upload, test set, manual)
- Confidence intervals
- Prediction visualization
- Results download

### 🔍 Project Scanner
- Scan single or multiple projects
- Analyze deep-timeseries usage patterns
- Feature usage statistics
- File-by-file breakdown
- Batch scanning capabilities
- Version detection

### ⚙️ Settings
- **User Management** (NEW!):
  - Change password
  - Add new users (admin only)
  - Delete users (admin only)
- UI preferences
- Training defaults
- Path configuration
- Usage analytics controls
- Settings import/export
- System information

## Installation

### Prerequisites

Ensure you have the deep-timeseries framework installed:

```bash
cd ..
pip install -e .
```

### Install Web UI Dependencies

```bash
cd web_ui
pip install -r requirements.txt
```

## Usage

### Starting the Dashboard

From the `web_ui` directory:

```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

### First Time Login

**Default Credentials:**
- Username: `admin`
- Password: `admin123`

**IMPORTANT**: Change your password immediately after first login!
1. Log in with default credentials
2. Navigate to "Settings"
3. Expand "User Management"
4. Use "Change Password" to set a secure password

### Quick Workflow

1. **Login**
   - Enter your username and password
   - Click "Login"

2. **Upload Dataset**
   - Go to "Dataset Manager"
   - Upload your CSV/Excel file or load an example
   - Preview and clean your data

2. **Configure Model**
   - Navigate to "Model Builder"
   - Select model type (LSTM/GRU/Transformer)
   - Adjust hyperparameters
   - Use presets or customize manually

3. **Train Model**
   - Go to "Training"
   - Select target column
   - Start training and monitor progress in real-time

4. **Evaluate Results**
   - Check "Results & Evaluation"
   - Analyze performance metrics
   - Review predictions vs. actual values

5. **Make Predictions**
   - Navigate to "Prediction"
   - Upload new data or use test set
   - Generate and download predictions

## Configuration

The UI stores settings in `~/.deep-timeseries/ui_settings.json`. You can:

- Modify settings through the Settings page
- Export settings for sharing
- Import settings from other users
- Reset to defaults

## Project Scanning

Use the Project Scanner to analyze how your projects use deep-timeseries:

```bash
# Via UI
Go to "Project Scanner" → Enter project path → Scan

# Via command line
python ../tools/scan_usage.py /path/to/project
```

## Customization

### Adding Custom Pages

Create a new file in `web_ui/pages/`:

```python
# web_ui/pages/my_page.py
import streamlit as st

def show():
    st.header("My Custom Page")
    # Your custom content here
```

Then add it to the navigation in `app.py`.

### Modifying Themes

Streamlit theming can be configured in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## Deployment

### Local Network Access

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Access from other devices: `http://<your-ip>:8501`

### Streamlit Cloud Deployment (Recommended)

Deploy to Streamlit Cloud for free hosting with automatic HTTPS:

**See [DEPLOYMENT.md](DEPLOYMENT.md) for complete step-by-step instructions.**

Quick steps:
1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Create new app pointing to `web_ui/app.py`
4. Deploy!

Your dashboard will be accessible from anywhere with a URL like:
`https://yourusername-deep-eagle-web-ui-app-xxxxxx.streamlit.app`

### Docker Deployment

Example Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Port Already in Use

```bash
streamlit run app.py --server.port 8502
```

### Missing Dependencies

```bash
pip install -r requirements.txt --upgrade
```

### Dataset Not Loading

- Ensure CSV has headers
- Check for encoding issues (try UTF-8)
- Verify file path is correct

### Model Training Errors

- Verify dataset is loaded
- Check model configuration
- Ensure GPU drivers are installed (for CUDA)

## Features Roadmap

- [x] Multi-user support ✅
- [x] User authentication ✅
- [ ] Real-time training (currently simulated)
- [ ] Experiment tracking integration
- [ ] Hyperparameter optimization UI
- [ ] Model comparison tool
- [ ] Custom callback editor
- [ ] Data augmentation interface
- [ ] Export to production formats (ONNX)
- [ ] Cloud storage integration
- [ ] Role-based access control (viewer/editor/admin)

## Contributing

To add new features to the UI:

1. Create feature in `pages/` directory
2. Add navigation link in `app.py`
3. Update this README
4. Test thoroughly

## Support

For issues specific to the Web UI:
- Check existing issues
- Provide browser console logs
- Include Streamlit version

For framework issues, see the main README.

## License

Same as deep-timeseries framework (MIT License)
