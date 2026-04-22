---
name: local-embeddings.instructions
description: Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.
---

# Instructions

- Prefer clarity, correctness, and maintainability over preserving legacy behavior.
- Breaking changes are allowed by default unless a task explicitly requires compatibility.
- Follow crate conventions in AGENTS.md (no implementation in lib.rs/mod.rs, group imports, explicit deps, etc.).

# Breaking-change checklist

- Update or add tests covering the new behavior.
- Update examples and documentation to reflect the new API/format.
