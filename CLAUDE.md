# General Prose guidelines

These should be strictly adhered to in both code, documentation, readmes, PR, commits. Any place you write anything.  

- Do not compose multiple sentences into one. 
- Do not use em dashes, you always end up composing multiple sentences into when you do this, in a prose which feels complicated
- Use bullet points generally for everything you do. 
  - Use nested bullet points to continue flow of the previous point instead of making it a single sentence, as done here.
- Be concise, but prefer simplicity over conciseness. 

# Coding and documentation guidelines

- Prefer small readable functions which do one thing. A function can be created to make code more readable even if the code block is not reused.
- Prefer code which conveys meaning using descriptive function and variable names. Don't write docstrings for every small function. Make sure you never simply write a docstring that restates what the function definition says
- Docstrings should talk about why/how of non-trivial parts. Don't state simple things. 
  - Err on the side of not documenting, your types and function names (the whole function signature) should carry intent
- Always prefer strong typing, along with descriptive type names for long type definitions. 
  - Pydantic is preferred over dataclasses since it gives easy JSON marshalling
  - Use jaxtyping for typing tensors
- Don't use matplotlib for non-interactive work (batch image saving, report generation, etc.) - it's slow for this. Prefer compositing with PIL directly.
  - Reserve matplotlib for interactive/exploratory plotting.

# Running experiments on Colab

- Use the `colab-experiments` skill. It holds the dependency rules, the notebook loop and how to verify a run.

# Parked work

- Persisting a neuron's density-peak forest was prototyped and parked, unreviewed.
  - It lives on branch `feat/cluster-fit`, draft PR #137.
  - If we ever want to persist the forest, look at that PR first.

# Formatting

- `scripts/git-hooks/pre-commit` sorts imports and runs `ruff format` on staged python files at commit time.
  - Install it once per clone with `ln -s ../../scripts/git-hooks/pre-commit .git/hooks/pre-commit`.
  - A partly staged file is only checked, so its unstaged hunks never get committed.
