#!/usr/bin/env python
"""
Simple PyPI upload helper script

Usage:
    python pypi_upload.py            # Upload to production PyPI
    python pypi_upload.py --test     # Upload to test PyPI first
    python pypi_upload.py --check    # Just validate package
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run command and print status"""
    print(f"\n📦 {description}...")
    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def main():
    project_dir = Path(__file__).parent
    dist_dir = project_dir / "dist"
    
    if not dist_dir.exists() or not list(dist_dir.glob("*")):
        print("❌ No distribution files found in dist/")
        print("   Run: python -m build")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 airbornehrs PyPI Upload Helper")
    print("=" * 60)
    
    # Check package
    print("\n✅ Checking package quality...")
    if not run_command(
        [sys.executable, "-m", "twine", "check", str(dist_dir / "*")],
        "Validating package"
    ):
        print("⚠️  Package validation had issues (non-fatal)")
    
    # Parse arguments
    test_mode = "--test" in sys.argv
    check_only = "--check" in sys.argv
    
    if check_only:
        print("\n✅ Package check complete!")
        sys.exit(0)
    
    # Upload
    if test_mode:
        print("\n⚠️  TEST MODE: Uploading to https://test.pypi.org")
        print("    (No credentials needed if configured in pip.ini)")
        cmd = [
            sys.executable, "-m", "twine", "upload",
            "--repository", "testpypi",
            str(dist_dir / "*")
        ]
        description = "Uploading to Test PyPI"
    else:
        print("\n🔐 PRODUCTION MODE: Uploading to https://pypi.org")
        print("    Make sure you have PyPI credentials configured!")
        print("    (Set in pip.ini or TWINE_USERNAME/TWINE_PASSWORD env vars)")
        
        response = input("\n⚠️  Continue with production upload? (type 'yes' to confirm): ")
        if response.lower() != "yes":
            print("❌ Upload cancelled")
            sys.exit(1)
        
        cmd = [
            sys.executable, "-m", "twine", "upload",
            str(dist_dir / "*")
        ]
        description = "Uploading to Production PyPI"
    
    if run_command(cmd, description):
        print("\n✅ Upload successful!")
        if test_mode:
            print("\n📥 To test installation from Test PyPI:")
            print("   pip install -i https://test.pypi.org/simple/ airbornehrs")
        else:
            print("\n📥 To install from PyPI:")
            print("   pip install airbornehrs")
    else:
        print("\n❌ Upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
