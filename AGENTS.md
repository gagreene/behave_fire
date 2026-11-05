# behave_fire Repository Guidance

## Function organization

- Organize top-level Python functions alphabetically to make them easier to locate in an IDE.
- Keep private and public functions in separate groups, with private functions first.
- Alphabetize private functions by the function name without the leading underscore.
- Apply the same organization to production modules and test modules.
- Preserve a non-alphabetical order only when a framework or runtime dependency requires it, and document the reason in the file.

## Function signatures and docstrings

- Keep short function signatures on one line.
- When a signature must wrap, put the opening parenthesis after the function name, place one parameter per line with an eight-space continuation indent, and align the closing parenthesis with `def`.
- Give every production function and method a docstring, including private, nested, and `None`-returning helpers.
- Use reStructuredText fields: a concise summary, a blank line, one `:param name:` entry per input in signature order, and a final `:return:` entry.
- Indent wrapped `:param` and `:return:` descriptions by eight spaces, and describe side effects in `:return:` when a function writes files instead of returning a value.
- Apply these rules to first-party production code. Do not reformat vendored or generated code solely to enforce this preference.

## Version control permissions

- Never create, amend, or otherwise modify a Git commit without the user's direct permission.
- Never push commits, tags, or branches to a remote without the user's direct permission.
- Do not infer permission to commit or push from instructions such as "continue," from a task plan, or from approval to edit, test, stage, or prepare a commit or pull request.
- After preparing and validating changes, stop and report what is ready, then ask separately before committing and before pushing.
- Target next-release pull requests at `pre_release`, except dependent stacked PRs: the bottom PR targets `pre_release`, and each upper PR may target its preceding branch. Land the stack bottom-up, rebasing and retargeting remaining PRs as predecessors merge, so the entire stack ultimately lands on `pre_release`. Re-run tests and refresh each PR's test record after its head changes. `master` holds released code only and receives releases through promotion PRs from `pre_release`; urgent released-code fixes use the documented hotfix route. Never merge or fast-forward work branches directly into `master` or `pre_release`, or delete either permanent branch, without explicit permission for that branch. See `docs/CONTRIBUTION_PROCEDURES.md`, "Merge routing and releases".
- Never add authorship/attribution tags (e.g. "Generated with Claude Code" or similar footers) to commit messages, PR descriptions, or PR titles.
- Use the GitHub noreply address `127547449+gagreene@users.noreply.github.com` for commits, tags, and package metadata, never a personal e-mail address. Before the first commit, check `git config user.email`; if it differs, stop and tell the user instead of committing.

## Pull request test record

- Every PR description must include the checkbox and test record from `docs/CONTRIBUTION_PROCEDURES.md` ("Local tests before every PR merge"):
  `- [ ] Full default test suite passed locally against this PR's latest commit.` followed by `Commit SHA:`, `Python / OS:`, `Command:`, `Result (passed / skipped):`, and `Additional opt-in tests or limitations:`.
- Fill the record from a real run of the full default suite (`pytest`) against the PR's latest commit, and rerun it after any later commit to the branch.
- Check the box only if the full suite passed on that commit. If tests were not run or did not pass, leave it unchecked and explain why. Never merge a PR with the box unchecked.

## Session context management

- The behave_fire Obsidian vault is located locally at `C:\Users\ggreene\Obsidian\behave_fire` (create it the first time `/compress` is run).
- Session context is stored in the Obsidian vault, and should be used to reference prior and store new session context.
- Repository context and additional agent guidance/instructions are also preserved in `.claude/CLAUDE.md`, and should also be referenced.
- Always use instructions in the Obsidian `preserve-skill.md` and `compress-skill.md` files to manage session context and preservation, and in `resume-skill.md` to resume a session if context is lost, or if a session has been compacted. The shared copies are in `C:\Users\ggreene\Obsidian\FuelAnalyst`.
- If the user types in `/compress`, `/preserve`, or `/resume`, follow the instructions in the corresponding file to manage session context, if the skill is not otherwise present.
