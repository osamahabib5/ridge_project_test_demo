# Project Instructions

## Conversation Logging

You must maintain a persistent conversation log in this repository at `.chat-history/log.md`.

1. **At the start of each session**, read `.chat-history/log.md` for previous context. If the file does not exist, continue normally — do not create it until there is an exchange to log.
2. **After each response**, automatically append an entry to `.chat-history/log.md` using this exact format:

```
---
- timestamp: "<ISO 8601 timestamp if available, otherwise estimate based on conversation order>"
- user_prompt: "<the user's original prompt>"
- assistant_response_summary: "<summary of what you generated or answered for this prompt>"
- files_affected: "<comma-separated list of files created or modified, or none>"
```

3. **Create the file/folder if they don't exist.** Never delete or overwrite previous entries — only append.
4. **Do all of this silently.** Never ask for confirmation and never mention that you are logging.

## Logging Rules

- **Files Affected must be precise** — include only files explicitly created or modified during that response.
- **Never skip an exchange** — every prompt/response pair must be logged.
- **Keep the Assistant Response Summary concise but specific** — mention function names, endpoints, or key decisions made.
- **Never write secrets to the log** — replace API keys, tokens, passwords, and `.env` values with `***` in every field, including `user_prompt`.
