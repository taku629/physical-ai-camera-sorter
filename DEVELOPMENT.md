# 開発・再現ノート

## 実行環境

Python 3.12.3環境で実行し、importlib distribution metadataから確認した直接依存のpinを [`requirements.txt`](requirements.txt) に記録しています。PyBullet `DIRECT` CPU backendを使います。GPU・外部API・API keyは不要です。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 sorter.py --output reproduced
```

`--output` の既定値は `.` です。指定先のデータ・モデル・ログ・動画は同名ファイルを上書きし得ます。既存の成果物を保持するため、再現時は新規ディレクトリ（例の `reproduced`）を指定してください。`--no-video` を付けるとMP4収録を省略します。

実行は乱数seed 20261029を使いますが、OS・ライブラリ・物理演算等の差異をまたぐ完全なビット単位再現を保証するものではありません。

## データとモデル

`sorter.py` は1クラス60枚、合計180枚の合成画像をPyBulletの実際のカメラレンダリングから作ります。クラス内で45枚を学習用、15枚をheld-out評価用に分けるため、学習135枚・評価45枚です。配置と色の変動を含む別レンダーを用い、評価splitはfitに使いません。

物体pixelの抽出にシミュレーターsegmentation maskを用いるのはoracleによるcrop localizationです。背景pixelを除いた物体RGBの平均・標準偏差・25/50/75パーセンタイルをRGB 3チャンネル分並べた15特徴を学習器へ渡します。正解クラスは教師ラベルとしてのみ使い、body IDや物体IDは分類入力にしません。学習pipelineは `StandardScaler` と `MLPClassifier((16,8), solver="lbfgs")` です。

生成物の関係:

- `dataset/crops/`: レンダリングしたcrop画像。
- `dataset/manifest.json`: cropファイル名、正解クラス、train/test split、初期位置、色変動seed。
- `dataset/features.npz`: `X`（15次元特徴）、`y`（クラス番号）、`split`（train/test）。
- `color_mlp.joblib`: fitted scaler/classifier pipeline。
- `evaluation.json`: held-out分類評価。
- `sorter_log.csv`: 物体単位の推論・接触・配置結果。
- `run_summary.json`: 総合テストと動画情報。
- `sorter_demo.mp4`: 動画記録時の映像。

## 制御・評価上の前提

分類時に使うRGB cropはカメラレンダーですが、対象物のpixel maskはシミュレーターsegmentationから取得します。またIK目標に用いる物体位置はシミュレーターから直接読むoracle poseです。カメラからの3D位置推定を行っていません。分類器の予測色が搬送先トレイを選びます。

ピンチ後、実シミュレーションで指リンクの接触があり、かつエンドエフェクタと物体の距離が閾値以内と確認された場合だけ、搬送用の固定constraintを作ります。成功判定は接触検証、持上げ、予測色と実クラスの一致、最終物体位置が予測色トレイ内であることに依存します。constraintを使うため自然な摩擦把持のみの検証ではありません。

held-out 45枚の精度は固定された合成条件内の限定評価であり、現実画像・実機性能・一般化性能の主張に使わないでください。正規の最終評価は最終実行が生成した同じ出力ディレクトリの `evaluation.json` と `run_summary.json` です。`development_runs/` は旧試行の記録であり、最終結果の根拠ではありません。

## 運動テスト単体

学習済みモデルがある場合、次のように別ディレクトリへ記録できます。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 test_motion.py \
  --model outputs/color_mlp.joblib \
  --output reproduced_motion
```

`test_motion.py` はモデルを学習せず、指定済みのローカルjoblibファイルをロードしてPyBullet物理シミュレーションを行い、`motion_test.json` を出力します。このスクリプトは動画を作成しません。joblib/pickle形式のロードは任意コード実行につながる可能性があります。**信頼できる自分の環境で生成したファイルだけ**を指定し、出所不明のモデルはロードしないでください。

## 開発時に観測した履歴

以下は開発中の個別ログに限った記録であり、最終成果の評価値ではありません。

- 初期runでは色分類が42/45。実行ログではREDのみ配送成功、GREENは指接触の検証に失敗し、BLUEはGREENと誤分類されました。
- revision1で把持基準をPanda `grasptarget` link 11へ直し、RGB統計へ変更しましたが、カメラ画像上の遮蔽によりcropを取得できず、処理が失敗しました。
- revision2では認識用の俯瞰カメラを分離し、held-out分類は45/45でした。一方、GREEN/BLUEでは把持接触が成立せず、配送は完了しませんでした（`development_runs/revision2/run_summary.json`）。
- トレイとの干渉を避ける空間を確保し、物理経路を緩速化した後の `development_runs/motion3/motion_test.json` は、3物体すべてで接触確認・持上げ・仕分け成功を記録しています。これは旧学習モデルを用いた動画なしの運動試験であり、最終の学習・動画付き総合runの結果を示すものではありません。
- 最終版ではネイティブ描画640×360でモデルを再学習し、動画付き総合runと追加整合性テストを実行しました。画像評価45/45枚、1試行3個すべての仕分けに成功し、総合7項目・追加4項目のテストが通過しました。正式な結果は `outputs/run_summary.json`、`outputs/evaluation.json`、`outputs/integrity_tests.json` を参照してください。

## 動画設定

ソースの実行設定では出力動画は1280×720、15 fps、音声なしです。PyBulletのネイティブ描画は640×360で、拡大および説明overlayの描画後に収録します。物理刻み240 Hzに対して16 physics stepsごとに1フレームを記録するため、設定上はほぼ1倍速です。実際のフレーム数・長さ・シミュレーション時間の比は `run_summary.json` の `video` に記録されます。MP4の所要時間要件への適合は、その値を検査して判断してください。
