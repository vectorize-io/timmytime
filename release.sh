#!/usr/bin/env bash
# Releases timmytime:  ./release.sh <X.Y.Z | patch | minor | major>
# Bumps the version in pyproject.toml, commits, tags v<version> and pushes. The release workflow then
# builds the package with the weights pinned in weights.lock, makes a GitHub release with the notes
# from CHANGELOG.md, and publishes to PyPI. Write the CHANGELOG.md entry first.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -eq 1 ] || { sed -n 2,5p "$0"; exit 1; }
CUR=$(grep -m1 '^version = ' pyproject.toml | cut -d'"' -f2)
IFS=. read -r MA MI PA <<< "$CUR"
case $1 in
  patch) V="$MA.$MI.$((PA + 1))" ;;
  minor) V="$MA.$((MI + 1)).0" ;;
  major) V="$((MA + 1)).0.0" ;;
  *) V=$1 ;;
esac
[[ $V =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "bad version: $V" >&2; exit 1; }
git diff --quiet && git diff --staged --quiet || { echo "commit or stash your changes first" >&2; exit 1; }
[ "$(git branch --show-current)" = main ] || { echo "release from main" >&2; exit 1; }
! git rev-parse -q --verify "refs/tags/v$V" >/dev/null || { echo "tag v$V exists" >&2; exit 1; }
grep -q "^## \[$V\]" CHANGELOG.md || { echo "add a '## [$V] - $(date +%F)' entry to CHANGELOG.md first" >&2; exit 1; }
if [ "$V" != "$CUR" ]; then
  sed -i.bak "s/^version = \".*\"/version = \"$V\"/" pyproject.toml && rm pyproject.toml.bak
  uv lock -q
  git commit -qam "Release $V"
fi
git tag -a "v$V" -m "Release $V"
git push -q && git push -q origin "v$V"
echo "v$V pushed: https://github.com/vectorize-io/timmytime/actions"
