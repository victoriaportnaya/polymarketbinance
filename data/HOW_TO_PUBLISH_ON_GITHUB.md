# GitHub: this project uses one repo at the root

The main remote is **https://github.com/victoriaportnaya/polymarketbinance**.

From the project root (`statistics/`):

```bash
git remote -v
git status
git push -u origin main
```

If you ever need a **data-only** fork, copy `data/` elsewhere and `git init` there separately.

## Large files

CSVs here are small. For huge raw zips, use [Git LFS](https://git-lfs.github.com/) or GitHub Releases.
