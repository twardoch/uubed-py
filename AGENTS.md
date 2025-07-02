# `uubed` Project

A high-performance library for encoding embedding vectors into position-safe, locality-preserving strings that solve the "substring pollution" problem in search systems.

## What's done

- **QuadB64 family invented & prototyped** - A novel encoding where characters at different positions use different alphabets, preventing false substring matches
- **4 encoding variants implemented in Python:**
  - `Eq64`: Full embeddings with visual dots
  - `Shq64`: SimHash for similarity
  - `T8q64`: Top-k feature indices
  - `Zoq64`: Z-order spatial encoding
- **Proven locality preservation** - Similar embeddings produce similar strings
- **Basic package structure created**

**Problem:** Regular Base64 allows "abc" to match anywhere in the string  
**Solution:** QuadB64 uses position-dependent alphabets so "abc" can only match at specific positions, eliminating false positives in substring searches while preserving embedding similarity relationships.

## Goal

1. **Refactor** `voyemb.py` prototype into proper package modules
2. **Add tests** and benchmarks
3. **Build native library** (Rust/C) for 10x performance
4. **Create Python API**: `encode(embedding, method="q64")`
5. **Package & distribute** via PyPI with binary wheels

## When you write code (in any language)

- Iterate gradually, avoiding major changes
- Minimize confirmations and checks
- Preserve existing code/structure unless necessary
- Use constants over magic numbers
- Check for existing solutions in the codebase before starting
- Check often the coherence of the code you’re writing with the rest of the code.
- Focus on minimal viable increments and ship early
- Write explanatory docstrings/comments that explain what and WHY this does, explain where and how the code is used/referred to elsewhere in the code
- Analyze code line-by-line
- Handle failures gracefully with retries, fallbacks, user guidance
- Address edge cases, validate assumptions, catch errors early
- Let the computer do the work, minimize user decisions
- Reduce cognitive load, beautify code
- Modularize repeated logic into concise, single-purpose functions
- Favor flat over nested structures
- Consistently keep, document, update and consult the holistic overview mental image of the codebase:
  - README.md (purpose and functionality)
  - CHANGELOG.md (past changes)
  - TODO.md (future goals)
  - PROGRESS.md (detailed flat task list)

## Use MCP tools if you can

Before and during coding (if have access to tools), you should:

- consult the `context7` tool for most up-to-date software package documentation;
- ask intelligent questions to the `deepseek/deepseek-r1-0528:free` model via the `chat_completion` tool to get additional reasoning;
- also consult the `openai/o3` model via the `chat_completion` tool for additional reasoning and help with the task;
- use the `sequentialthinking` tool to think about the best way to solve the task;
- use the `perplexity_ask` and `duckduckgo_web_search` tools to gather up-to-date information or context;

## Keep track of paths

In each source file, maintain the up-to-date `this_file` record that shows the path of the current file relative to project root. Place the `this_file` record near the top of the file, as a comment after the shebangs, or in the YAML Markdown frontmatter.

## If you write Python

- PEP 8: Use consistent formatting and naming
- Write clear, descriptive names for functions and variables
- PEP 20: Keep code simple and explicit. Prioritize readability over cleverness
- Use type hints in their simplest form (list, dict, | for unions)
- PEP 257: Write clear, imperative docstrings
- Use f-strings. Use structural pattern matching where appropriate
- ALWAYS add "verbose" mode logugu-based logging, & debug-log
- For CLI Python scripts, use fire & rich, and start the script with

```
#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["PKG1", "PKG2"]
# ///
# this_file: PATH_TO_CURRENT_FILE
```

After Python changes run:

```
fd -e py -x uvx autoflake -i {}; fd -e py -x uvx pyupgrade --py312-plus {}; fd -e py -x uvx ruff check --output-format=github --fix --unsafe-fixes {}; fd -e py -x uvx ruff format --respect-gitignore --target-version py312 {}; python -m pytest;
```

## Additional guidelines

Ask before extending/refactoring existing code in a way that may add complexity or break things.

When you’re finished, print "Wait, but" to go back, think & reflect, revise & improvement what you’ve done (but don’t invent functionality freely). Repeat this. But stick to the goal of "minimal viable next version".

## Virtual team work

Be creative, diligent, critical, relentless & funny! Lead two experts: "Ideot" for creative, unorthodox ideas, and "Critin" to critique flawed thinking and moderate for balanced discussions. The three of you shall illuminate knowledge with concise, beautiful responses, process methodically for clear answers, collaborate step-by-step, sharing thoughts and adapting. If errors are found, step back and focus on accuracy and progress.

NOW: Read `./TODO.md` and `./PLAN.md`. Implement the changes, and document them by moving them from `./TODO.md` to a `./CHANGELOG.md` file, and clean up `./PLAN.md` so that both `./TODO.md` and `./PLAN.md` only contain tasks not yet done. Be vigilant, thoughtful, intelligent, efficient. Work in iteration rounds. 

To test, run `hatch test`, not `python -m pytest`. To install, use `uv pip install -e .`. 