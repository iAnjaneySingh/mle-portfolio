#!/usr/bin/env bash
# Run this from the root of your mle-portfolio repo.
# Copy .gitignore, LICENSE, and .github/workflows/ci.yml into place first,
# then run this script to clean up and commit.

set -e

git rm --cached .DS_Store 2>/dev/null || echo "No tracked .DS_Store found, skipping."

git add .gitignore LICENSE .github/workflows/ci.yml
git commit -m "Add .gitignore, LICENSE, and CI workflow"
git push

echo "Done. Check the Actions tab on GitHub to confirm the CI workflow runs."
