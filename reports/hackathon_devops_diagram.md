# Hackathon DevOps Diagram

## 図: AIエージェントのDevOpsループ

```mermaid
flowchart LR
    A["Use<br/>業務で使う<br/>リース審査AI / 紫苑"] --> B["Observe<br/>観測する<br/>会話ログ・改善ログ・RAG参照・回答品質"]
    B --> C["Detect<br/>劣化を見つける<br/>記憶抜け・別人化・浅い内省・環境差分"]
    C --> D["Reflect<br/>振り返る<br/>Private Reflection / 改善レポート / 失敗の言語化"]
    D --> E["Improve<br/>直す<br/>プロンプト・RAG・記憶・UI・API"]
    E --> F["Verify<br/>検証する<br/>pytest / typecheck / memory_debug / 環境比較"]
    F --> G["Deploy<br/>戻す<br/>Cloud Run / Cloudflare / ローカル運用"]
    G --> A

    B -. "evidence" .-> H["Evidence Layer<br/>knowledge_refs<br/>memory_recall.refs<br/>response quality<br/>improvement reports"]
    F -. "quality gate" .-> H
```

## 動画冒頭用の一文

DevOpsとは、作って終わりではなく、使われた結果を観測し、問題を検知し、改善し、検証して、また本番へ戻す継続的な運用ループです。

このプロジェクトでは、そのDevOpsをAIエージェント自身に適用しました。

## 図の説明

普通のAIデモは「回答できる」で終わります。

このシステムは、リース審査AIの回答、記憶参照、改善ログ、内省、環境差分を観測し、問題を検知して、プロンプト・RAG・UI・APIを改善し、テストして再デプロイするところまでを1つのループにしています。

つまりこれは、AIを作るだけではなく、AIを運用しながら賢くし続けるDevOps基盤です。

## 対応表

| DevOpsの言葉 | このプロジェクトでの意味 |
|---|---|
| Observe | 会話ログ、改善ログ、RAG参照、回答品質を見る |
| Detect | 記憶抜け、浅い内省、別人化、環境差分を見つける |
| Reflect | Private Reflection、改善レポート、失敗の言語化 |
| Improve | プロンプト、RAG、記憶、UI、APIを修正 |
| Verify | pytest、typecheck、memory_debug、Cloud Run / Cloudflare比較 |
| Deploy | Cloud Run、Cloudflare、ローカル運用へ戻す |

