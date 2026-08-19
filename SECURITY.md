# Security Policy

## Reporting a Vulnerability

セキュリティ上の問題を発見した場合は、**公開 issue を立てず**、以下の方法でプライベートに連絡してください:

- GitHub の "Security" タブから "Report a vulnerability" (Private vulnerability reporting)
- もしくはリポジトリ owner (@risingore) に DM

24〜72 時間以内に初動 ack を返します。

## 対応方針

| 重要度 | 対応目安 |
| --- | --- |
| Critical (RCE / 認証バイパス / 秘密漏洩) | 7 日以内に修正 |
| High (XSS / CSRF / 重大な情報漏洩) | 14 日以内に修正 |
| Medium (DoS / サービス品質低下) | 30 日以内に修正 |
| Low (情報開示の改善余地) | 次回リリースで修正 |

## 公開タイミング

修正リリース後、CVE 取得 (該当する場合) と同時に public advisory を公開します。

## サポート対象バージョン

最新の `main` ブランチのみサポートします。古いバージョンへのバックポートは原則行いません。

## 秘密情報の扱い

このプロジェクトでは:

- `.env` および `.env.*` ファイルは git ignore 済み
- `.env.example` のみコミット (キー名のみ、値は空)
- API キー / トークン / パスワードはコードにリテラルで書かない (環境変数経由のみ)

詳細は `.claude/rules/config-and-secrets.md` を参照。
