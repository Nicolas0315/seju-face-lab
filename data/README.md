# Data Directory / data ディレクトリ

**EN**: Place consent-cleared reference images in `data/raw/`.

**JA**: 許諾済みの参照画像は `data/raw/` に置きます。

```text
data/raw/
  reference_001.jpg
  reference_002.jpg
```

For subject-folder reviews:

人物別レビューを行う場合:

```text
data/subjects/
  subject_a/
    image1.jpg
  subject_b/
    image1.jpg
```

`data/raw/`, `data/processed/`, `data/subjects/`, and `outputs/` are ignored by Git.
Do not commit private images, generated portraits, extracted vectors, or SNS intermediate data.

`data/raw/`、`data/processed/`、`data/subjects/`、`outputs/` は Git 管理外です。
個人画像、生成画像、抽出ベクトル、SNS中間データをコミットしないでください。
