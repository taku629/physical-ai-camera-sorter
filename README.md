# 画像認識AIによる荷物仕分けロボット

PyBullet上のFranka Pandaアームが、カメラ画像から小箱の色を分類し、予測色に対応した物理トレイへ運ぶデモです。CPU上のシミュレーションであり、実機ではありません。

> コード作成・実行・説明資料作成にはAIエージェントの補助を使用しました。利用者本人による実機実験や学習体験を記録したものではありません。

## 記録済みの結果

| 確認項目 | 最終実行の実測値 |
|---|---|
| 学習画像 / 未使用の評価画像 | 135枚 / 45枚（3色各15枚を評価） |
| 画像分類 | 45/45枚正解 |
| 物理仕分け | 1試行・3個すべて成功 |
| 動画 | 72.13秒、1280×720、15 fps、音声なし、ほぼ1倍速 |
| 総合テスト / 追加整合性テスト | 7/7項目 / 4/4項目通過 |

正式な根拠は [`outputs/evaluation.json`](outputs/evaluation.json)、[`outputs/run_summary.json`](outputs/run_summary.json)、[`outputs/sorter_log.csv`](outputs/sorter_log.csv)、[`outputs/integrity_tests.json`](outputs/integrity_tests.json) です。MP4のメタデータは [`outputs/video_probe.json`](outputs/video_probe.json) に保存しています。[`report.md`](report.md) は日本語の説明資料です。

## AIと制御のつながり

PyBullet TinyRendererで3色各60枚、合計180枚の小箱画像を実際に描画します。位置・向き・色に小さな変動を加え、135枚を学習、45枚を学習に使わない評価用へ分けます。

対象物の画素はシミュレーターの正解segmentation maskで切り出します。背景を除いたRGBの平均・標準偏差・25/50/75パーセンタイル、合計15特徴を `StandardScaler` と `MLPClassifier(hidden_layer_sizes=(16,8), solver="lbfgs")` へ渡します。物体IDや正解クラスは分類入力ではありません。

**実行時の予測色が搬送先トレイを選びます。** 物体の位置はシミュレーターの既知座標を使用し、IKと関節位置制御で動かします。指と物体の接触および距離を確認した場合に限り、固定constraintによる仮想把持補助を作成します。解放後の物理計算と中心位置に基づき仕分け成否を判定します。

追加テストでは、同じ物体IDの見た目だけを別の色へ交換すると予測もその色に変わることと、未接触状態では把持補助を拒否することを確認しました。

## 限界

固定された合成環境、3色の立方体、既知の配置に限定された評価です。45枚の精度は現実環境への汎化性能ではありません。対象画素の切り出しにはoracle mask、IK目標にはoracle poseを使っており、物体検出や画像からの3D位置推定は未実装です。搬送には仮想把持補助を使い、純粋な摩擦把持、実機、汎用物体認識、強化学習、未知の障害物への対応を実証していません。

## 再現する

Python 3.12.3環境で実行しました。直接依存の実環境バージョンは [`requirements.txt`](requirements.txt) に固定しています。GPU・外部APIキーは不要です。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 sorter.py --output reproduced
```

同じ出力先のファイルは上書きされるため、記録済みの `outputs/` ではなく、新しい出力先を指定してください。動画不要なら `--no-video` を追加できます。OSやライブラリの差をまたぐビット単位の再現は保証しません。

```bash
# 保存済みモデル・分割・未接触拒否・色入替の検査（約数秒）
OPENBLAS_NUM_THREADS=1 python3 test_integrity.py --artifacts outputs --output reproduced_integrity.json

# 学習を省いた物理動作テスト（動画なし）
OPENBLAS_NUM_THREADS=1 python3 test_motion.py --model outputs/color_mlp.joblib --output reproduced_motion
```

joblib/pickleのロードは任意コード実行につながるため、出所不明のモデルをロードしないでください。この検査では同梱の、ローカルで実際に学習したモデルのみを使用します。

## ファイル構成

| 場所 | 内容 |
|---|---|
| `sorter.py` | シーン、学習、推論、物理制御、収録、実測判定 |
| `test_integrity.py` / `test_motion.py` | 独立した整合性検査 / 運動試験 |
| `outputs/dataset/crops/` | 実際に描画した学習・評価crop 180枚 |
| `outputs/dataset/manifest.json` / `features.npz` | クラス・分割・位置・色変動seed / 15次元特徴 |
| `outputs/color_mlp.joblib` | 実際に学習したモデル |
| `outputs/` のJSON・CSV | 正式な評価と動作ログ |
| `outputs/sorter_demo.mp4` | 実行動画 |
| `development_runs/` | 成功へ書き換えていない旧試行の記録 |

ネイティブ描画は640×360で、説明文字のため1280×720へ拡大して収録しています。物理計算240 Hz、16ステップごとに1フレームでほぼ1倍速です。映像は生成AIではなく実際のシミュレーターの連続出力です。

開発経緯は [`DEVELOPMENT.md`](DEVELOPMENT.md)、既存モデル・ライセンスの確認先は [`THIRD_PARTY.md`](THIRD_PARTY.md) を参照してください。Pandaモデルは `pybullet_data` からロードし、メッシュ等をこのコード一式へ複製していません。独自コード用のライセンスは未設定です。
