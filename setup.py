import os
import platform
import subprocess
import sys


def run(cmd):
    """Run a shell command and stream output."""
    print(f"→ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("Command failed. Stopping.")
        sys.exit(1)


def main():
    system = platform.system().lower()

    print(f"Detected OS: {system}")

    # ---- Step 1: Create virtual environment ----
    print("\nCreating virtual environment (.venv)...")
    run(f"{sys.executable} -m venv .venv")

    # ---- Step 2: Determine venv paths ----
    if system == "windows":
        pip_path = r".venv\Scripts\pip.exe"
        activate_cmd = r".venv\Scripts\Activate.ps1"
    else:
        pip_path = ".venv/bin/pip"
        activate_cmd = "source .venv/bin/activate"

    # ---- Step 3: Upgrade pip ----
    print("\nUpgrading pip inside the virtual environment...")
    run(f"{pip_path} install --upgrade pip")

    # ---- Step 4: Install requirements ----
    if not os.path.exists("requirements.txt"):
        print("No requirements.txt found. Skipping package installation.")
    else:
        print("\nInstalling packages from requirements.txt...")
        run(f"{pip_path} install -r requirements.txt")

    # ---- Step 5: Print activation instructions ----
    print("\nSetup complete!")
    print("To activate the virtual environment, run:\n")

    if system == "windows":
        print(f"    {activate_cmd}")
    else:
        print(f"    {activate_cmd}")

    print("\nAfter activation, run python normally:")
    print("    python your_script.py")


if __name__ == "__main__":
    main()
