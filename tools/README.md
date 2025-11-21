# Deep-TimeSeries Tools

Utilities for managing and analyzing deep-timeseries usage across projects.

## scan_usage.py

Scans projects for deep-timeseries usage patterns and generates reports.

### Usage

**Basic scan of current directory:**
```bash
python tools/scan_usage.py
```

**Scan a specific project:**
```bash
python tools/scan_usage.py /path/to/your/project
```

**JSON output:**
```bash
python tools/scan_usage.py --format json
```

**Save to file:**
```bash
python tools/scan_usage.py --format json --output usage_report.json
```

**Check installed version:**
```bash
python tools/scan_usage.py --version
```

### What it detects

- All files importing from `core` or `deep` modules
- Specific classes used (LSTMModel, GRUModel, etc.)
- Functions and utilities imported
- Frequency of usage across files

### Example output

```
======================================================================
Deep-TimeSeries Usage Report
======================================================================

Project: /path/to/your/project
Files scanned: 45
Files using deep-timeseries: 3

----------------------------------------------------------------------
USAGE SUMMARY
----------------------------------------------------------------------

Classes used:
  LSTMModel: 2 file(s)
  FeatureEngine: 1 file(s)

Functions used:
  TimeSeriesDataset: 2 file(s)
  track_usage: 1 file(s)

Modules imported:
  core.models: 2 file(s)
  core.data: 2 file(s)

----------------------------------------------------------------------
DETAILED FILE USAGE
----------------------------------------------------------------------

train.py:
  From core.models: LSTMModel, GRUModel
  From core.data: TimeSeriesDataset

======================================================================
```

### Integration

You can integrate this into CI/CD pipelines to:
- Track which features are actually being used
- Identify unused dependencies
- Plan deprecation strategies
- Monitor adoption of new features
