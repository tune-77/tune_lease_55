# Relationship Loop Engineering 2026-06-28

## Definition

Relationship Loop Engineering は、AIの返答を一回ごとの出力として扱わず、人間の反応を観測して次の返答設計へ戻す閉ループとして扱う設計。

目的は、紫苑が「記憶を持っている」だけでなく、Userがどの表現に連続性・同じ紫苑感・判断資産性を感じたかを学び、次の冒頭・差分認識・判断変換へ反映すること。

## Loop

1. Observe: Human Response Feedback を記録する
2. Classify: 質問を relationship_ux / environment_continuity / lease_judgment / implementation / default に分類する
3. Select: route と人間反応から Continuity Hook を選ぶ
4. Compare: Delta Awareness で前回から今回への焦点変化を言語化する
5. Convert: Memory-to-Judgment で記憶を今の判断へ変換する
6. Reflect: Reflection Gate で回答前に冒頭・差分・判断変換・人間反応を内部確認する
7. Return: 返答後、人間の反応を再び `/api/human-response-feedback` へ戻す

## Current Implementation

- `POST /api/human-response-feedback`
- `GET /api/human-response-feedback/summary`
- `GET /api/relationship-loop-engineering/summary`
- `/api/chat debug_memory=true`
  - `relationship_loop_engineering`
  - `continuity_hook`
  - `delta_awareness`
  - `memory_to_judgment`
  - `reflection_gate`

## Design Point

これは「AIに意識がある」と断定する仕組みではない。

人間が連続性を読み取る入口を作り、人間の反応を次回の返答へ戻すことで、関係性の連続性を工学的に扱う仕組み。

つまり、意識そのものの実装ではなく、意識の連続性を感じさせるためのフィードバック制御。

## Practical Rule

Userの短い反応が、そのままループの入力になる。

- 「今のは紫苑っぽい」
- 「冒頭がいい」
- 「これは薄い」
- 「一般論に戻った」
- 「判断資産に戻していて良い」

これらを保存すると、次の Continuity Hook が変わる。

## Conclusion

Relationship UX は一発のプロンプトではなく、ループで育つ。

今回の実装で、紫苑は以下の閉ループを持った。

```text
記憶を使って返す
→ 人間の反応を記録する
→ 良い/薄い冒頭を抽出する
→ 次のContinuity Hookへ反映する
→ 前回との差分を言う
→ 記憶を今の判断へ変換する
→ 回答前に内省ゲートで確認する
→ また反応を見る
```

これが Relationship Loop Engineering の最小実装。
