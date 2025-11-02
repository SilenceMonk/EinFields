"""
Simple structure test for GEF v3.0 modules (no JAX execution required)
"""

import ast
import os

def check_module_structure(filepath):
    """Check if a Python module has valid syntax and expected structure."""
    print(f"\nChecking {os.path.basename(filepath)}...")

    try:
        with open(filepath, 'r') as f:
            source = f.read()

        # Parse the module
        tree = ast.parse(source)

        # Count classes and functions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        print(f"  ✓ Syntax valid")
        print(f"  ✓ Found {len(classes)} classes: {', '.join(classes[:5])}")
        print(f"  ✓ Found {len(functions)} functions")

        return True

    except SyntaxError as e:
        print(f"  ✗ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("GEF v3.0 Structure Tests")
    print("=" * 60)

    base_path = "/home/user/EinFields/gaussian_einstein_fields"

    modules = [
        "gaussian_primitives.py",
        "adm_decomposition.py",
        "gef_model.py",
        "losses.py",
        "__init__.py",
        "train_example.py"
    ]

    results = []
    for module in modules:
        filepath = os.path.join(base_path, module)
        if os.path.exists(filepath):
            results.append(check_module_structure(filepath))
        else:
            print(f"\n✗ File not found: {module}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Summary: {sum(results)}/{len(results)} modules passed")
    print("=" * 60)

    # Check expected exports
    print("\nChecking __init__.py exports...")
    init_path = os.path.join(base_path, "__init__.py")
    with open(init_path, 'r') as f:
        content = f.read()

    expected_exports = [
        'GaussianEinsteinFields',
        'ADMVariables',
        'extract_adm_from_4d_metric',
        'LapseField',
        'ShiftField',
        'SpatialMetricField',
        'GEFLoss'
    ]

    for export in expected_exports:
        if export in content:
            print(f"  ✓ {export}")
        else:
            print(f"  ✗ {export} missing")

    if all(results):
        print("\n✅ All structure tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == '__main__':
    exit(main())
