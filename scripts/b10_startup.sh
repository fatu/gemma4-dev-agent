if [ "{{workdir}}" != /testbed ]; then cd / && rmdir /testbed 2>/dev/null; ln -sfn {{workdir}} /testbed; [ "$(readlink /testbed)" = "{{workdir}}" ] || exit 1; fi
cd /testbed || exit 1
if [ "__STRIP_REFS__" = 1 ] && git rev-parse HEAD >/dev/null 2>&1; then
  git for-each-ref --no-merged HEAD --format='delete %(refname)' | git update-ref --stdin
  rm -f "$(git rev-parse --git-dir)/FETCH_HEAD" "$(git rev-parse --git-dir)/ORIG_HEAD"
  git reflog expire --expire=now --all >/dev/null 2>&1
  timeout 40 git gc -q --prune=now >/dev/null 2>&1
  [ "$(git for-each-ref --no-merged HEAD | wc -l)" -eq 0 ] || exit 1
fi
true
