# AGENTS for `uubed-py` (Python Code)

This repository contains the Python implementation and API for the `uubed` library.

## Role of this Repository:
- **Python Package Development:** Manages the Python package structure, modules, and dependencies.
- **Python API:** Implements the high-level `encode()` and `decode()` functions, providing a user-friendly interface.
- **FFI Bindings:** Handles the integration with the native Rust library via PyO3.
- **Python-specific Testing & Benchmarking:** Develops and runs tests and benchmarks relevant to the Python layer.

## Key Agents and Their Focus:
- **Python Developer:** Implements and maintains the Python codebase, ensuring adherence to PEP standards and Pythonic practices.
- **API Designer:** Focuses on creating an intuitive and efficient Python API for `uubed`.
- **Performance Engineer (Python):** Optimizes Python-specific code and ensures efficient data transfer between Python and Rust.

If you work with Python, use 'uv pip' instead of 'pip', and use 'uvx hatch test' instead of 'python -m pytest'. DO NOT USE `python -m pytest` !

When I say /report, you must: Read all `./TODO.md` and `./PLAN.md` files and analyze recent changes. Document all changes in `./CHANGELOG.md`. From `./TODO.md` and `./PLAN.md` remove things that are done. Make sure that `./PLAN.md` contains a detailed, clear plan that discusses specifics, while `./TODO.md` is its flat simplified itemized `- [ ]`-prefixed representation. When I say /work, you must work in iterations like so: Read all `./TODO.md` and `./PLAN.md` files and reflect. Work on the tasks. Think, contemplate, research, reflect, refine, revise. Be careful, curious, vigilant, energetic. Verify your changes. Think aloud. Consult, research, reflect. Then update `./PLAN.md` and `./TODO.md` with tasks that will lead to improving the work you’ve just done. Then '/report', and then iterate again.