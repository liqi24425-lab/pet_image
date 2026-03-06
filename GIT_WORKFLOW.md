# Git Workflow Policy

This repository follows a strict delivery workflow.

## Baseline
- Primary branch: `main`
- Primary remote: `origin` -> `https://github.com/liqi24425-lab/pet_image.git`

## Per-Task Delivery Rule
For each user request that changes code or tracked files:
1. Make only the requested changes.
2. Run minimal required verification (syntax check or core command run).
3. Create exactly one commit for that request.
4. Push immediately to `origin/main`.

## Conflict Handling
If push is rejected:
1. `git fetch origin`
2. `git rebase origin/main`
3. Resolve conflicts
4. Push again

## Commit Message Convention
Use:
- `type(scope): summary`

Examples:
- `fix(inference): correct test path fallback`
- `chore(repo): initialize repository and main workflow`

The summary must explain both what changed and why.
