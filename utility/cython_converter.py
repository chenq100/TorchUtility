import sys
import shutil
import os
from Cython.Build import cythonize
from setuptools import setup, Extension

def compile_pyx(source_py):
    if not source_py.endswith('.py'):
        print("Error: The file must be a .py file")
        sys.exit(1)

    base_name = source_py[:-3]
    pyx_file = f"{base_name}.pyx"

    shutil.copy(source_py, pyx_file)
    print(f"Copied {source_py} to {pyx_file}")

    try:
        # Compile .pyx to .so
        setup(
            ext_modules=cythonize([Extension(base_name, [pyx_file])], 
                                  compiler_directives={'language_level': "3"}),
            script_args=['build_ext', '--inplace']
        )
        print(f"Compiled {pyx_file} to {base_name}.so successfully")
    except Exception as e:
        print(f"Compilation failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(pyx_file):
            os.remove(pyx_file)
            print(f"Deleted {pyx_file}")

        c_file = f"{base_name}.c"
        if os.path.exists(c_file):
            os.remove(c_file)
            print(f"Deleted {c_file}")

        if os.path.exists('build'):
            shutil.rmtree('build')
            print("Deleted build directory")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cython_converter.py <filename.py>")
        sys.exit(1)

    source_py = sys.argv[1]
    compile_pyx(source_py)

