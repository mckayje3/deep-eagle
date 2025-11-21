"""Project scanner integration page"""

import streamlit as st
import subprocess
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def show():
    """Display project scanner page"""

    st.header("🔍 Project Scanner")

    st.markdown("""
    Scan your projects to analyze how they use deep-timeseries.
    This helps you understand feature usage, plan upgrades, and identify dependencies.
    """)

    st.markdown("---")

    # Scanner configuration
    st.subheader("📂 Select Project to Scan")

    scan_method = st.radio(
        "Choose method:",
        ["Enter Path", "Browse Recent", "Scan Multiple"]
    )

    scan_results = None

    if scan_method == "Enter Path":
        project_path = st.text_input(
            "Project path:",
            placeholder="C:\\Users\\...\\my_project",
            help="Enter the full path to the project directory"
        )

        col1, col2 = st.columns([1, 3])

        with col1:
            scan_button = st.button("🔍 Scan Project", type="primary")

        with col2:
            check_version = st.checkbox("Check installed version", value=True)

        if scan_button and project_path:
            scan_results = run_scanner(project_path, check_version)

    elif scan_method == "Browse Recent":
        # Store recent scans in session state
        if 'recent_scans' not in st.session_state:
            st.session_state.recent_scans = []

        if st.session_state.recent_scans:
            selected = st.selectbox(
                "Select a recent scan:",
                st.session_state.recent_scans,
                format_func=lambda x: x.get('project_path', 'Unknown')
            )

            if st.button("View Results"):
                scan_results = selected
        else:
            st.info("No recent scans available. Scan a project to get started!")

    else:  # Scan Multiple
        st.markdown("### Batch Scan Multiple Projects")

        paths_input = st.text_area(
            "Enter project paths (one per line):",
            height=150,
            help="Enter multiple project paths, one per line"
        )

        if st.button("🔍 Scan All Projects") and paths_input:
            paths = [p.strip() for p in paths_input.split('\n') if p.strip()]

            progress_bar = st.progress(0)
            status_text = st.empty()

            all_results = []

            for i, path in enumerate(paths):
                status_text.text(f"Scanning {i+1}/{len(paths)}: {path}")
                result = run_scanner(path, check_version=True)
                if result:
                    all_results.append(result)
                progress_bar.progress((i + 1) / len(paths))

            if all_results:
                st.success(f"✅ Scanned {len(all_results)} projects")
                show_batch_results(all_results)

    # Display scan results
    if scan_results:
        display_scan_results(scan_results)


def run_scanner(project_path, check_version=False):
    """Run the project scanner tool"""

    try:
        # Build command
        cmd = [
            'python',
            'tools/scan_usage.py',
            project_path,
            '--format', 'json'
        ]

        if check_version:
            cmd.append('--version')

        # Run scanner
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            scan_data = json.loads(result.stdout)

            # Save to recent scans
            if 'recent_scans' not in st.session_state:
                st.session_state.recent_scans = []

            # Add to recent scans (keep last 10)
            st.session_state.recent_scans.insert(0, scan_data)
            st.session_state.recent_scans = st.session_state.recent_scans[:10]

            return scan_data
        else:
            st.error(f"Scanner error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        st.error("Scanner timed out. Project may be too large.")
        return None
    except Exception as e:
        st.error(f"Error running scanner: {e}")
        return None


def display_scan_results(results):
    """Display scan results"""

    st.markdown("---")
    st.subheader("📊 Scan Results")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Files Scanned", f"{results['files_scanned']:,}")

    with col2:
        st.metric("Files Using Deep", results['files_using_deep'])

    with col3:
        unique_classes = len(results['summary']['classes'])
        st.metric("Unique Classes", unique_classes)

    with col4:
        if 'installed_version' in results:
            st.metric("Version", results['installed_version'])
        else:
            st.metric("Version", "Not detected")

    st.markdown("---")

    # Usage breakdown
    if results['files_using_deep'] > 0:
        st.subheader("📈 Feature Usage")

        col1, col2 = st.columns(2)

        with col1:
            # Classes usage
            if results['summary']['classes']:
                classes_df = pd.DataFrame([
                    {'Feature': k, 'Files': v}
                    for k, v in sorted(
                        results['summary']['classes'].items(),
                        key=lambda x: -x[1]
                    )
                ])

                fig_classes = px.bar(
                    classes_df,
                    x='Feature',
                    y='Files',
                    title='Classes Used',
                    color='Files',
                    color_continuous_scale='Blues'
                )
                fig_classes.update_layout(height=400)
                st.plotly_chart(fig_classes, use_container_width=True)

        with col2:
            # Functions usage
            if results['summary']['functions']:
                funcs_df = pd.DataFrame([
                    {'Feature': k, 'Files': v}
                    for k, v in sorted(
                        results['summary']['functions'].items(),
                        key=lambda x: -x[1]
                    )
                ])

                fig_funcs = px.bar(
                    funcs_df,
                    x='Feature',
                    y='Files',
                    title='Functions Used',
                    color='Files',
                    color_continuous_scale='Greens'
                )
                fig_funcs.update_layout(height=400)
                st.plotly_chart(fig_funcs, use_container_width=True)

        # Detailed file-by-file breakdown
        st.markdown("---")
        st.subheader("📁 File-by-File Breakdown")

        for file_path, usage in results['usage_by_file'].items():
            with st.expander(f"📄 {file_path}"):
                col1, col2 = st.columns(2)

                with col1:
                    if usage['classes']:
                        st.markdown("**Classes:**")
                        for cls in usage['classes']:
                            st.markdown(f"- {cls}")

                with col2:
                    if usage['functions']:
                        st.markdown("**Functions:**")
                        for func in usage['functions']:
                            st.markdown(f"- {func}")

                if usage['from_imports']:
                    st.markdown("**Imports:**")
                    for module, items in usage['from_imports'].items():
                        st.code(f"from {module} import {', '.join(items)}")

    else:
        st.info("This project does not appear to use deep-timeseries")

    # Export results
    st.markdown("---")
    st.subheader("💾 Export Scan Results")

    col1, col2 = st.columns(2)

    with col1:
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="scan_results.json",
            mime="application/json"
        )

    with col2:
        # Create summary CSV
        summary_data = []
        for file_path, usage in results['usage_by_file'].items():
            summary_data.append({
                'File': file_path,
                'Classes': ', '.join(usage['classes']),
                'Functions': ', '.join(usage['functions'])
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="scan_summary.csv",
                mime="text/csv"
            )


def show_batch_results(all_results):
    """Show aggregated results from multiple projects"""

    st.markdown("---")
    st.subheader("📊 Batch Scan Summary")

    # Aggregate statistics
    total_files = sum(r['files_scanned'] for r in all_results)
    total_using = sum(r['files_using_deep'] for r in all_results)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Projects Scanned", len(all_results))

    with col2:
        st.metric("Total Files", f"{total_files:,}")

    with col3:
        st.metric("Files Using Deep", total_using)

    # Feature usage across all projects
    all_classes = {}
    all_functions = {}

    for result in all_results:
        for cls, count in result['summary']['classes'].items():
            all_classes[cls] = all_classes.get(cls, 0) + count

        for func, count in result['summary']['functions'].items():
            all_functions[func] = all_functions.get(func, 0) + count

    col1, col2 = st.columns(2)

    with col1:
        if all_classes:
            classes_df = pd.DataFrame([
                {'Class': k, 'Usage Count': v}
                for k, v in sorted(all_classes.items(), key=lambda x: -x[1])
            ])

            fig = px.bar(
                classes_df,
                x='Class',
                y='Usage Count',
                title='Classes Usage Across All Projects'
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if all_functions:
            funcs_df = pd.DataFrame([
                {'Function': k, 'Usage Count': v}
                for k, v in sorted(all_functions.items(), key=lambda x: -x[1])
            ])

            fig = px.bar(
                funcs_df,
                x='Function',
                y='Usage Count',
                title='Functions Usage Across All Projects'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Project comparison
    st.markdown("---")
    st.subheader("📊 Project Comparison")

    comparison_data = []
    for result in all_results:
        comparison_data.append({
            'Project': Path(result['project_path']).name,
            'Files Scanned': result['files_scanned'],
            'Files Using Deep': result['files_using_deep'],
            'Unique Classes': len(result['summary']['classes']),
            'Unique Functions': len(result['summary']['functions'])
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
