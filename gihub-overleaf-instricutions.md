# Overleaf ↔ GitHub ↔ Local (VSCode) Report Sync

This documents how the `report/` folder (LaTeX report) is linked to Overleaf, and how
to push/pull changes between VSCode, GitHub, and Overleaf.

## Why a separate GitHub repo

Overleaf's "GitHub Sync" links a whole GitHub repo root to a whole Overleaf project
root. Since `report/` is just one subfolder inside the `ProjektAnlagenring` monorepo
(alongside `maps/`, `routing/`, `ABsurveys/`, etc.), it can't be pointed at a subfolder
directly. Instead:

- A dedicated, separate GitHub repo holds just the report:
  https://github.com/MiguelUrenaPliego/ProjektVerkehr-Report
- That repo is linked to the Overleaf project via Overleaf's GitHub Sync.
- The `report/` folder inside this monorepo is linked to that same GitHub repo using
  `git subtree`, so it stays a normal folder in this repo's history while also being
  syncable to/from the standalone repo.

Chain: **local `report/` (this repo) ⇄ GitHub (`ProjektVerkehr-Report`) ⇄ Overleaf project**

## How `report/` relates to the main GitHub repo (subtree, not submodule)

Your main code repo (`ProjektAnlagenring` on GitHub) contains everything: `maps/`,
`routing/`, `ABsurveys/`, and `report/`. `report/` is a completely normal folder in
that repo's history — it's just files, tracked and committed like anything else.
There is **no `.gitmodules` file, no submodule, no nested `.git` folder** inside
`report/`. If you clone `ProjektAnlagenring` fresh, `report/` shows up already
populated, same as any other folder — nothing extra to initialize.

The trick is `git subtree`, which is different from a submodule:

- A **submodule** stores a *pointer* to a commit in another repo; the folder is
  technically a separate repo nested inside, and needs `git submodule init/update`
  to actually have files in it.
- A **subtree** copies the *actual files and history* of another repo's content
  directly into your repo's own history, in a subfolder. There's nothing to
  initialize — the files are just there, like normal.

The link to the separate `ProjektVerkehr-Report` GitHub repo isn't stored as
metadata in `ProjektAnlagenring` at all. It only exists as:

1. A git remote you configured locally: `report-repo` → points at
   `https://github.com/MiguelUrenaPliego/ProjektVerkehr-Report.git`. This remote is
   a setting in your local `.git/config`, not something committed/pushed — anyone
   else cloning `ProjektAnlagenring` would need to add this remote themselves if
   they want to sync `report/` with Overleaf.
2. Shared commit history: because `report/`'s files were originally pushed to
   `ProjektVerkehr-Report` via `git subtree split`/`push`, some old commits are
   shared between the two repos' histories. That shared history is what lets
   `git subtree push`/`pull` figure out what's new on each side and merge only the
   diff, instead of overwriting everything every time.

So: `ProjektVerkehr-Report` isn't a "sub-repo" in any git-native sense (no
submodule link exists) — it's an independent repo that happens to share file
history with the `report/` subfolder of `ProjektAnlagenring`, and the `subtree`
commands are what translate changes between the two.

## One-time setup (already done)

1. Created empty private GitHub repo `ProjektVerkehr-Report`.
2. Added it as a git remote in this repo:
   ```bash
   git remote add report-repo https://github.com/MiguelUrenaPliego/ProjektVerkehr-Report.git
   ```
3. Split `report/`'s history into that remote and linked it as a subtree:
   ```bash
   git subtree add --prefix=report report-repo main --squash
   ```
4. In Overleaf: Menu → **GitHub** → linked the project to `ProjektVerkehr-Report`
   (branch `main`).

You shouldn't need to repeat this setup unless the remote or subtree link is broken.

## Day-to-day sync

### VSCode/local → GitHub → Overleaf

1. Edit files under `report/` locally as normal, commit them to this repo like any
   other change:
   ```bash
   git add report/...
   git commit -m "..."
   ```
2. Push just the `report/` subfolder to the standalone GitHub repo:
   ```bash
   git subtree push --prefix=report report-repo main
   ```
3. In Overleaf: Menu → **GitHub** → click **Sync** to pull the GitHub changes into
   the Overleaf project.

### Overleaf → GitHub → VSCode/local

1. Edit the project directly in Overleaf.
2. In Overleaf: Menu → **GitHub** → **Sync** (this pushes Overleaf's changes as a
   commit to the `ProjektVerkehr-Report` GitHub repo).
3. Pull those changes back into this repo:
   ```bash
   git fetch report-repo
   git subtree pull --prefix=report report-repo main --squash
   ```

## Notes / gotchas

- Overleaf's GitHub import renamed `report.tex` to `main.tex` (its default main-file
  convention) the first time the project was linked. Keep using `main.tex` as the
  entry point going forward.
- `git subtree push`/`pull` only work correctly once the folder has been registered
  via `git subtree add` (not just a plain folder with matching file contents). If you
  ever see `fatal: can't squash-merge: 'report' was never added`, the link is broken
  and needs to be re-established:
  ```bash
  git rm -r --cached report
  rm -rf report
  git commit -m "Remove report/ to re-link as subtree"
  git subtree add --prefix=report report-repo main --squash
  ```
  This is safe as long as `report-repo/main`'s content matches (or you're okay
  overwriting local with) what's currently in Overleaf/GitHub.
- `report.md` at the repo root is unrelated scratch content, not part of this sync,
  and is gitignored (`/report.md` in `.gitignore`).
