# Coding and documentation guidelines

- Prefer small readable functions which do one thing. A function can be created to make code more readable even if the code block is not reused.
- Prefer code which conveys meaning using descriptive function and variable names. Don't write docstrings for every small function. Make sure you never simply write a docstring that restates what the function definition says
- Docstrings should talk about why/how of non-trivial parts. Don't state simple things. 
- Prefer bullet points. Don't merge sentences into more complex sentences which are harder to read.
- Always prefer strong typing, along with descriptive type names for long type definitions. 
  - Pydantic is preferred over dataclasses since it gives easy JSON marshalling
  - Use jaxtyping for typing tensors
- Don't use matplotlib for non-interactive work (batch image saving, report generation, etc.) - it's slow for this. Prefer compositing with PIL directly.
  - Reserve matplotlib for interactive/exploratory plotting (notebook `show_*` helpers).
