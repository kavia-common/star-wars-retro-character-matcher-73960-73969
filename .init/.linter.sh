#!/bin/bash
cd /home/kavia/workspace/code-generation/star-wars-retro-character-matcher-73960-73969/swcg_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

