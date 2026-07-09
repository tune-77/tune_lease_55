#!/bin/bash
# 2026-07-08 棚卸しで「中身が master 取り込み済み」と確認したリモートブランチ34本を
# 削除する一回限りのスクリプト（PR #538 の続き・Claude セッションで監査）。
#
# 安全装置: 監査時点の SHA と一致するブランチだけ削除する。
# 棚卸し後に新しいコミットが積まれたブランチは [skip] して手動判断に委ねる。
#
# 使い方（Mac のリポジトリで）:
#   bash scripts/cleanup_stale_branches_20260708.sh          # dry-run（削除しない）
#   bash scripts/cleanup_stale_branches_20260708.sh --apply  # 実削除
set -uo pipefail

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1
[ $APPLY -eq 1 ] && echo "=== 実削除モード ===" || echo "=== dry-run（--apply で実削除） ==="

deleted=0; skipped=0
while read -r expected branch; do
    [ -z "$branch" ] && continue
    actual=$(git ls-remote origin "refs/heads/$branch" | cut -f1)
    if [ -z "$actual" ]; then
        echo "[skip] $branch: 既に存在しない"; skipped=$((skipped+1)); continue
    fi
    if [ "$actual" != "$expected" ]; then
        echo "[skip] $branch: 監査後に更新あり（要手動確認）"; skipped=$((skipped+1)); continue
    fi
    if [ $APPLY -eq 1 ]; then
        if git push origin --delete "$branch"; then
            echo "[del]  $branch"; deleted=$((deleted+1))
        else
            echo "[fail] $branch: 削除失敗"; skipped=$((skipped+1))
        fi
    else
        echo "[dry]  $branch: 削除対象（SHA一致）"; deleted=$((deleted+1))
    fi
done <<'BRANCH_LIST'
b1c8d67a951d04448b42a5384a895030a9bc1ee9 claude/chromadb-cloudrun-connection-il9mzz
90a6baa6d6bebba421f3f5215671ff088f135679 claude/fable-hackathon-strategy-nj7and
470fb85039bea1d4eed42098564cb11322cf957b claude/lease-system-improvements-c0vhrg
75736399b125b724c7ddb8c18b3c703ff6135fb1 claude/shion-memory-accuracy-5lvotj
b15f48dd53bf2b2d566b2d1bb5ec90f124b53823 feat/error-alert-slack
1ce4ce9c6aca1f8935a5f6f80f1b5b4b8743da0a feat/execute-codex-queue-gemini-fallback
7557af310d0030ea118e023e67265ed81e7ce9f9 feat/hotstart-min-instances
8c4b990efb97793a7a17590fdb25a44b58b6096d feat/morning-report-slack
57a3aeb205ea88837f660762679f240d684834b3 feat/rev-077-dialogue-memory-update
28ed450c0e0c195820834cda0f489c892654c2d9 feat/rev-078-shion-yesterday-grumble
eca086a0e37b912f3bbb7c3b2901525ecaf80868 feat/rev-080-shion-self-audit-feedback
8bb725dbb121ca6c28166d2e4c742f3aca623297 feat/rev-083-dynamic-keyword-extraction
e189bc47b1dc4ce443a4d62fde7818ec3bda4f2e feat/rev-085-fix-gemini-model-and-grumble-loop
97f8ad053b064fd3ba745ba90061442af8472f91 feat/rev-093-remove-privacy-field
340765f239f21f37ea32cc3523055f97d119dde3 feat/rev-094-private-reflection
5855137247db81dc4c3657ea5700593c116c394e feat/rev-103b-appliers
1ee1e086a140d763ab963427e1dd883d2fd2a199 feat/rev-104-build-codex-queue-batch-apply-guard
4edd32528ee3af75c7730a773c74343e84a968bd feat/rev-108-pipeline-health-auto-fix
5626db33cd37e4abfb2bd995db75fcdd638f2792 feat/rev-139-demo-mode
7fb176538b0091cc9512725aee4bb5e9e27e1dc3 feat/rev-145-chat-history-limit-expand
34337244f057deb773be211f9d41075d739774c3 feat/rev-155-central-feedback
8185f0a697972b1c6bedab749602ace2a87e97f2 feat/rev-156-reflection-central
a2b3577e829eb0d192ff0fa623167970ef1aec7d feat/rev-159-fastapi-db-connection
ccd588d8b77ce52fe5a18fa0b86726023d8d111f feat/rev-163-icloud-to-gcs-sync
8390f83447d55d32f667b124f2ce2f7d43cc37a6 feat/rev-169-sidebar-cleanup
c35da537a26a5307594f616bb19485cb67fd94d4 feat/rev-170-sidebar-ux
7f74f026dfd7245dc936076bd82e1468045e15dc feat/rev-171-system-overview-update
c440f5359d29ecee2ebe91fa19fa8b700e692cc4 feat/rev-172-ocr-dashboard
dafa82122260bca1409abd3fa7bfc2556de7f41e feat/rev-173-vad-fix
f72ce64ad295dbc77bd26720fa3bed92b6916d23 feat/slack-dispatch-notifier
d5546acad377fe31aa09afa4242a9cae85135942 fix/auto-apply-test-bypass-for-low-risk
0252859de54ed9e7b262264adfcca1d0cf914374 fix/rev-046-wizard-500-error
5e128735c672c3fa50a714c9f4dd6ab000950922 fix/rev-103b-scoring-weight-connect
59f1e7e3bb3b1392ad39712cfdc5a7c4910eec44 fix/rev-126-cloud-run-db-and-vault
BRANCH_LIST

echo ""
echo "対象: $deleted 本 / スキップ: $skipped 本"
[ $APPLY -eq 0 ] && echo "実削除するには --apply を付けて再実行してください。"
