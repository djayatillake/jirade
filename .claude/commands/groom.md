Groom Jira ticket $ARGUMENTS by analyzing requirements and producing an implementation plan.

## Workflow

1. **Fetch the ticket and all comments** using `jirade_get_issue` with key=$ARGUMENTS

2. **Check for an existing grooming plan** — scan comments for one starting with `[jirade grooming] Implementation Plan:`. If found, check whether a subsequent comment from someone other than jirade exists. Read that reply and use your judgement to determine if it's an approval (e.g. "approved", "lgtm", "looks good", "go ahead", thumbs up, or any affirmative) or feedback/rejection.

3. **If the plan is approved** — tell the user the plan has been approved and suggest they run `process-ticket` which will automatically pick up the plan context.

4. **If the plan was posted but got feedback** — revise the plan incorporating the feedback and post the updated plan via `jirade_add_comment` with the `[jirade grooming] Implementation Plan:` prefix.

5. **If questions were previously asked and answered** — incorporate the answers into your understanding of the requirements.

6. **Explore the codebase** — use Read, Grep, Glob tools to understand the relevant code. For dbt models, also use `jirade_analyze_deprecation` and `jirade_generate_schema_docs` if applicable.

7. **Decide next action:**
   - If requirements are unclear or ambiguous → post clarifying questions to Jira via `jirade_add_comment` with `[jirade grooming] Questions:` prefix. Tell the user you've posted questions and they should re-run `/groom $ARGUMENTS` after the requestor answers.
   - If requirements are clear → post the implementation plan via `jirade_add_comment` with `[jirade grooming] Implementation Plan:` prefix. Tell the user you've posted a plan and they should re-run `/groom $ARGUMENTS` after the requestor reviews it.

## Plan Structure

When posting an implementation plan, use this structure in the Jira comment:

```
[jirade grooming] Implementation Plan:

## Summary of Changes
<Brief description of what will be implemented>

## Files to Modify / New Files
- `path/to/file.sql` — description of changes
- `path/to/new_file.sql` — (new) description

## Impact Analysis
<For dbt models: downstream models affected, breaking changes, etc.>
<For other code: components affected, API changes, etc.>

## Implementation Steps
1. Step one
2. Step two
3. ...

## Testing Strategy
- How to verify the changes work correctly

## Risks / Open Questions
- Any remaining concerns

## Estimated Scope
Small / Medium / Large

---
Reply to this comment to approve or provide feedback on this plan.
```

## Important Rules

- All grooming comments MUST use the `[jirade grooming]` prefix so they're identifiable
- Do NOT block or wait for replies — post your questions/plan and complete the skill invocation
- When re-invoked, pick up from the comment history — this is an exit-and-resume interaction model
- Use the existing jirade MCP tools (`jirade_get_issue`, `jirade_add_comment`, `jirade_analyze_deprecation`, `jirade_generate_schema_docs`) for Jira and dbt operations
