#!/usr/bin/env python3
"""
Dependency Scanner for deep-timeseries
Scans projects for usage patterns and version information.
"""

import ast
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional


class DeepUsageScanner(ast.NodeVisitor):
    """AST visitor to detect deep-timeseries imports and usage"""

    def __init__(self):
        self.imports: Set[str] = set()
        self.classes_used: Set[str] = set()
        self.functions_used: Set[str] = set()
        self.from_imports: Dict[str, List[str]] = defaultdict(list)

    def visit_Import(self, node):
        """Visit import statements"""
        for alias in node.names:
            if 'core' in alias.name or 'deep' in alias.name:
                self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit from...import statements"""
        if node.module and ('core' in node.module or 'deep' in node.module):
            for alias in node.names:
                self.from_imports[node.module].append(alias.name)
                if alias.name.endswith('Model') or alias.name in ['FeatureEngine', 'Trainer']:
                    self.classes_used.add(alias.name)
                else:
                    self.functions_used.add(alias.name)
        self.generic_visit(node)


def scan_file(file_path: Path) -> Optional[DeepUsageScanner]:
    """Scan a single Python file for deep-timeseries usage"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        scanner = DeepUsageScanner()
        scanner.visit(tree)

        # Only return if we found deep-timeseries usage
        if scanner.imports or scanner.from_imports:
            return scanner
        return None
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return None


def scan_project(project_path: Path, output_format: str = 'text') -> Dict:
    """Scan entire project for deep-timeseries usage"""
    results = {
        'project_path': str(project_path),
        'files_scanned': 0,
        'files_using_deep': 0,
        'usage_by_file': {},
        'summary': {
            'classes': defaultdict(int),
            'functions': defaultdict(int),
            'modules': defaultdict(int),
        }
    }

    # Find all Python files
    python_files = list(project_path.rglob('*.py'))
    results['files_scanned'] = len(python_files)

    for py_file in python_files:
        scanner = scan_file(py_file)
        if scanner:
            results['files_using_deep'] += 1
            rel_path = py_file.relative_to(project_path)

            results['usage_by_file'][str(rel_path)] = {
                'imports': list(scanner.imports),
                'from_imports': dict(scanner.from_imports),
                'classes': list(scanner.classes_used),
                'functions': list(scanner.functions_used),
            }

            # Update summary
            for cls in scanner.classes_used:
                results['summary']['classes'][cls] += 1
            for func in scanner.functions_used:
                results['summary']['functions'][func] += 1
            for module in scanner.imports:
                results['summary']['modules'][module] += 1
            for module in scanner.from_imports:
                results['summary']['modules'][module] += 1

    # Convert defaultdicts to regular dicts for JSON serialization
    results['summary']['classes'] = dict(results['summary']['classes'])
    results['summary']['functions'] = dict(results['summary']['functions'])
    results['summary']['modules'] = dict(results['summary']['modules'])

    return results


def get_installed_version(project_path: Path) -> Optional[str]:
    """Try to detect installed deep-timeseries version"""
    # Check for pip freeze or requirements.txt
    try:
        import subprocess
        result = subprocess.run(
            ['pip', 'show', 'deep-timeseries'],
            capture_output=True,
            text=True,
            cwd=project_path
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass

    return None


def print_text_report(results: Dict):
    """Print a human-readable text report"""
    print("\n" + "=" * 70)
    print(f"Deep-TimeSeries Usage Report")
    print("=" * 70)
    print(f"\nProject: {results['project_path']}")
    print(f"Files scanned: {results['files_scanned']}")
    print(f"Files using deep-timeseries: {results['files_using_deep']}")

    if results['files_using_deep'] > 0:
        print("\n" + "-" * 70)
        print("USAGE SUMMARY")
        print("-" * 70)

        if results['summary']['classes']:
            print("\nClasses used:")
            for cls, count in sorted(results['summary']['classes'].items(), key=lambda x: -x[1]):
                print(f"  {cls}: {count} file(s)")

        if results['summary']['functions']:
            print("\nFunctions used:")
            for func, count in sorted(results['summary']['functions'].items(), key=lambda x: -x[1]):
                print(f"  {func}: {count} file(s)")

        if results['summary']['modules']:
            print("\nModules imported:")
            for module, count in sorted(results['summary']['modules'].items(), key=lambda x: -x[1]):
                print(f"  {module}: {count} file(s)")

        print("\n" + "-" * 70)
        print("DETAILED FILE USAGE")
        print("-" * 70)

        for file_path, usage in results['usage_by_file'].items():
            print(f"\n{file_path}:")
            if usage['imports']:
                print(f"  Imports: {', '.join(usage['imports'])}")
            if usage['from_imports']:
                for module, items in usage['from_imports'].items():
                    print(f"  From {module}: {', '.join(items)}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Scan projects for deep-timeseries usage patterns'
    )
    parser.add_argument(
        'project_path',
        type=Path,
        nargs='?',
        default=Path.cwd(),
        help='Path to project to scan (default: current directory)'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--version',
        action='store_true',
        help='Also check installed version'
    )

    args = parser.parse_args()

    if not args.project_path.exists():
        print(f"Error: Path '{args.project_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not args.project_path.is_dir():
        print(f"Error: Path '{args.project_path}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Scan the project
    results = scan_project(args.project_path, args.format)

    # Optionally check version
    if args.version:
        version = get_installed_version(args.project_path)
        results['installed_version'] = version

    # Output results
    if args.format == 'json':
        output = json.dumps(results, indent=2)
        if args.output:
            args.output.write_text(output)
        else:
            print(output)
    else:
        if args.output:
            import io
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            print_text_report(results)
            sys.stdout = old_stdout
            args.output.write_text(buffer.getvalue())
        else:
            print_text_report(results)

        if args.version and results.get('installed_version'):
            print(f"Installed version: {results['installed_version']}")


if __name__ == '__main__':
    main()
