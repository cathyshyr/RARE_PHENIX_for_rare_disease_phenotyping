import subprocess
import sys
from pathlib import Path


def run_step(name, command):
    print("=" * 80)
    print(f"Running: {name}")
    print("=" * 80)
    subprocess.run(command, check=True)


def main():
    repo_root = Path(__file__).resolve().parents[1]

    module1 = repo_root / "scripts" / "smoke_test_module1_hf.py"
    module2 = repo_root / "scripts" / "smoke_test_module2_lightweight.py"

    if not module1.exists():
        raise FileNotFoundError(f"Missing Module 1 smoke test: {module1}")
    if not module2.exists():
        raise FileNotFoundError(f"Missing Module 2 smoke test: {module2}")

    run_step(
        "Module 1 HF adapter smoke test",
        [sys.executable, str(module1)],
    )

    run_step(
        "Module 2 lightweight HPO standardization smoke test",
        [sys.executable, str(module2)],
    )

    print("=" * 80)
    print("SUCCESS: all quickstart smoke tests passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
