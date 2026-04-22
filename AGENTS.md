# AGENTS.MD

## Setup & Commands

- **Package Manager**: Use `pnpm`. Never use `npm` or `yarn`.
- **Install Dependencies**: `pnpm install`.
- **Development**: `pnpm dev`.
- **Build**: `pnpm build`.
- **Test**: `pnpm test`.

## Code Style & Standards

- **TypeScript**: Use strict mode. Prefer functional patterns over classes.
- **Formatting**: Single quotes, no semicolons.
- **Linting**: Run `pnpm lint` before every commit.
- **Commits**: Follow [Conventional Commits](https://github.com/pnpm/pnpm/blob/main/AGENTS.md) (e.g., `feat:`, `fix:`).

## Repository Guardrails

- **Critical**: Do not modify files in `dist/` or `node_modules/` manually.
- **Security**: Never include hardcoded API keys or secrets.
- **Dependencies**: Use `pnpm add <package>` for new dependencies; do not edit `package.json` manually for versioning.
