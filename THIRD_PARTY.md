# 第三者ソフトウェア・アセット

このプロジェクトは以下の直接依存を使用します。バージョンは実行環境の `importlib.metadata` から取得し、[`requirements.txt`](requirements.txt) に固定しています。ライセンス欄は同じ環境にインストールされたdistribution metadataで確認した値です。依存の再配布・改変時には、各配布物に含まれるライセンス本文と現行条件を改めて確認してください。

| Distribution | 固定バージョン | metadata上のライセンス | 公式確認先 |
| --- | --- | --- | --- |
| PyBullet (`pybullet`) | 3.2.7 | zlib | [Bullet Physics GitHub](https://github.com/bulletphysics/bullet3) |
| NumPy (`numpy`) | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | [NumPy source](https://github.com/numpy/numpy) |
| scikit-learn (`scikit-learn`) | 1.9.1 | BSD-3-Clause | [scikit-learn source](https://github.com/scikit-learn/scikit-learn) |
| joblib | 1.6.0 | BSD-3-Clause | [joblib source](https://github.com/joblib/joblib) |
| ImageIO (`imageio`) | 2.38.1 | BSD-2-Clause | [ImageIO source](https://github.com/imageio/imageio) |
| imageio-ffmpeg | 0.7.0 | BSD-2-Clause | [imageio-ffmpeg on PyPI](https://pypi.org/project/imageio-ffmpeg/) |
| opencv-python-headless | 5.0.0.93 | Apache 2.0 | [OpenCV-Python on PyPI](https://pypi.org/project/opencv-python-headless/) |

requirementsのインストールにより、これらが必要とする追加の推移依存も導入されます。それらの全パッケージ・ライセンス一覧を固定または網羅したSBOMではありません。配布前には使用環境に対して別途ライセンス・依存監査を行ってください。

## Franka Pandaモデル

シミュレーターは `pybullet_data` から `franka_panda/panda.urdf` をロードします。モデルの配布元は[Bullet3内のURDF](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf)、同じディレクトリの[LICENSE.txt](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_data/franka_panda/LICENSE.txt)にはApache License 2.0が記載されています。このリポジトリはそのモデルやメッシュを同梱していません。モデル利用・再配布時には配布元のライセンスと同梱通知を確認してください。

PyBulletのAPI資料: [PyBullet Quickstart Guide](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)。

## 独自コード

このリポジトリには独自コード用のライセンスファイルや権利許諾は設定されていません。この文書は独自コードに第三者ライセンスを付与するものではありません。利用・再配布の許諾が必要な場合は権利者に確認してください。
