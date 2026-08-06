# Cursorへの新しい指令（詳細版）

## 依頼内容
添付されたPDF (`new_project_spec.pdf`) に基づいて、新しいプログラムを実装してください。
PDFの解析結果を以下に記載しますので、これに従って実装を進めてください。

## 制約
- **本体（既存の `lease_logic_sumaho10.py` など）とは切り離した、別のプログラムとして作成すること**。
- `lease_scoring_system/` というディレクトリを新規作成し、その中で完結させてください。

## 実装タスク一覧
1. **音響効果の実装**: Tone.jsを使用して、素数企業（マウスオーバー）、承認/否決（クリック）などに音をつける。
2. **実データ連携システム**: Pythonスクリプト (`export_visualization_data.py`) でJSONデータを出力し、HTML側で読み込む。
3. **Tune Spaceタブの作成**: 審査データを使った「音楽的ビジュアライゼーション空間」の実装。

## ファイル構成予定
```
lease_scoring_system/
├── export_visualization_data.py  (Pythonスクリプト: データ出力用)
├── visualization_extended.html   (拡張版HTML: 音響+実データ+TuneSpace)
└── data/
    └── results.json              (生成されるデータファイル)
```

## 具体的な実装ステップ
1. `lease_scoring_system/` ディレクトリを作成。
2. `export_visualization_data.py` を作成（PDF内のコードを参照）。
3. `visualization_extended.html` を作成し、既存の「数学的多様体」コードをベースに、Tone.jsの読み込みと新機能（実データfetch, Tune Space描画）を追加。

実装完了後、以下のコマンドで実行できるようにしてください：
```bash
cd lease_scoring_system
python export_visualization_data.py
# その後、visualization_extended.html をブラウザで開く
```

以上、よろしくお願いします。
