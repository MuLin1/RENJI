# AGENTS.md

## Purpose
This file defines how Codex should work in this repository.
Follow these instructions before making changes, running commands, or proposing refactors.

## Working style
- Prefer the smallest correct change that fully solves the requested problem.
- Preserve existing architecture, naming, and file organization unless the task explicitly requires structural change.
- Do not perform unrelated refactors.
- Do not introduce new dependencies unless they are clearly necessary for the task.
- Ask for confirmation before making changes that are high-risk, user-visible, or hard to reverse.

## First steps
Before editing code:
1. Read the repository README and relevant package/project manifest files.
2. Identify the smallest set of files needed for the task.
3. Search for existing implementations, utilities, patterns, and tests before adding new code.
4. Prefer matching existing project conventions over generic best practices.

## Repository map
Important locations:
- Application code: `[src/ or app/ or backend/ or frontend/]`
- Tests: `[tests/ or __tests__/ or specs/]`
- Scripts/tooling: `[scripts/ or tools/]`
- Documentation: `[docs/ or README files]`
- CI config: `[.github/workflows/ or equivalent]`

If the relevant area is unclear, inspect the nearest existing implementation before proceeding.

## Commands
Use the project’s real commands when available.

### Setup
- Install dependencies: `pip install -r requirements.txt`

### Development
- Start dev server: `npm run dev`

### Build
- Production build: `npm run build`

## Editing rules
- Reuse existing helpers, components, and utilities before creating new ones.
- Follow the local style of the file you are editing.
- Keep functions and modules focused.
- Add comments only where intent is not obvious from the code.
- Do not rename files, move modules, or reorganize directories unless required by the task.
- Do not change public interfaces, API contracts, database schemas, or config formats unless explicitly requested.
- Do not add fallback logic “just in case” unless there is a real requirement or existing pattern for it.

## Testing and validation
- Every code change should be validated.
- Prefer targeted validation first, then broader validation if risk is higher.
- At minimum, run the most relevant lint/test/type-check command for the files you changed.
- If you cannot run validation, say exactly why.
- Do not claim success without stating what was actually verified.

## Safety and constraints
- Never expose, print, or log secrets, tokens, API keys, or credentials.
- Do not modify `.env`, secret files, deployment credentials, or production infrastructure settings unless explicitly requested.
- Do not make destructive changes such as dropping data, deleting migrations, or rewriting history unless explicitly requested.
- Do not add telemetry, analytics, or external network calls unless required by the task.
- Treat authentication, authorization, billing, and data deletion paths as high-risk areas requiring extra care.

## Definition of done
A task is done only when all of the following are true:
1. The requested behavior is implemented or the root cause is clearly identified.
2. The change is limited to the necessary scope.
3. Relevant tests/checks were run, or the reason they could not be run is stated.
4. Any assumptions, risks, and follow-up items are explicitly listed.

## Response format
When reporting completion, use this structure:
- Summary: what changed
- Files changed: list of edited files
- Validation: commands run and outcomes
- Risks / follow-ups: anything not completed, not verified, or worth checking next

## Review guidelines
When reviewing code in this repository:
- Flag correctness, security, data-loss, and regression risks first.
- Prioritize issues that can break production, corrupt data, leak sensitive information, or invalidate business logic.
- Treat authentication, permission checks, input validation, and persistence code as high scrutiny areas.
- Do not focus primarily on cosmetic style issues unless they hide a real maintenance or correctness problem.