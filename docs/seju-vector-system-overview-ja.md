# Seju顔ベクトル化・分析・機械学習 全体仕様（日本語レビュー版）

更新日: 2026-08-25  
対象仕様: `docs/superpowers/specs/2026-08-25-seju-vector-decomposition-candidate-search-design.md`  
対象実装: `seju-face-lab` のローカル研究パイプライン

## 1. 目的と結論

このシステムは、Seju公式サイトのローカル保存画像だけを対象に、顔画像をベクトル化し、個人ごとの撮影差を抑えたうえで、Seju画像集合に共通する局所的な構造を分析する。

主要な出力は次の二つである。

1. **Seju近似スコア**: クリーンなSejuローカル画像集合の中心性と部分空間内包性を、LOSO（1人物を丸ごと除外する交差検証）で校正した0〜100の局所百分位。
2. **架空候補探索**: 近似スコアだけを最大化せず、安定成分内の支持、画像品質、候補間多様性、匿名レビュー結果を別軸で保持する候補集合。

このスコアは所属確率、本人照合、魅力度、人格、人口統計、人気予測ではない。「人気」は顔ベクトルから推定せず、架空候補に対するブラインド比較で別に測る。

## 2. 現行（As-Is）フロー

現行コードは、画像収集、複数バックエンドでのベクトル化、平均・中央値モデル、生成画像評価、各種レポートを持つ。一方、データゲート、モデル契約、人物分離LOSO、106点幾何、近似スコアの校正は一つの強制パイプラインになっていない。

```mermaid
flowchart TD
    A[Seju公式サイト] --> B[sources.py\n候補URL収集]
    B --> C[download_manifest.jsonl\n画像とSHA-256]
    C --> D[subject_vectors.py\n人物フォルダ単位]
    C --> E[model.py\n画像集合単位]
    D --> F[backends.py\nInsightFace / deterministic等]
    E --> F
    F --> G[画像ベクトル]
    G --> H[画像平均または人物均等中心]
    H --> I[既存evaluate / audit / compare]
    I --> J[CSV・JSON・Markdownレポート]
    H --> K[既存generation]
    K --> L[生成画像]
    L --> I

    X[未強制: signature除外] -.-> C
    Y[未実装: 106点幾何] -.-> G
    Z[未実装: LOSO校正スコア] -.-> I
```

### 現行監査で確認済みの状態

- 35プロフィールから265候補、成人条件を満たす34プロフィール259ファイル。
- 259ファイルは欠損0、SHA-256不一致0、URLホストはすべて`seju.tokyo`。
- signature画像34枚が現行モデルに混入。
- InsightFaceはsignature 34枚と通常画像1枚を棄却し、224枚を採用。
- 33件のサムネイル候補は、知覚ハッシュとSSIMによる近重複判定が未完了。

したがって、現行のベクトル生成成功は確認できるが、現行中心ベクトルを「Seju顔モデル合格」とは扱わない。

## 3. 目標（To-Be）全体フローチャート

```mermaid
flowchart TD
    A[公式ソースマニフェスト] --> B{出自・SHA-256・成人条件}
    B -- 不合格 --> R1[棄却理由を記録]
    B -- 合格 --> C{signature / 非写真の事前除外}
    C -- 不合格 --> R1
    C -- 合格 --> D{顔検出ゲート\n顔数=1・姿勢・品質}
    D -- 不合格 --> R2[顔ゲート棄却記録]
    D -- 合格 --> E[整列済み顔crop]
    E --> F1[512次元 neural vector\nunit norm]
    E --> F2[106点 landmark\ngeometry vector]
    E --> G{人物内の近重複\npHash + SSIM}
    G -- 重複 --> R3[最高品質以外を抑止]
    G -- 独立画像 --> H[人物別 robust template]
    F1 --> H
    F2 --> H
    H --> I{人物数>=20\n各人物>=3独立画像}
    I -- 不足 --> DIAG[診断モデルのみ]
    I -- 合格 --> J[人物均等 spherical center]
    J --> K[tangent-space robust components]
    K --> L[LOSO校正]
    L --> M[Seju近似スコア\n中心百分位 + manifold百分位]
    L --> N{B1対C1 promotion gate}
    N -- C1不合格 --> O[B1人物均等基準を採用]
    N -- C1合格 --> P[C1 robust modelを採用]
    O --> Q[架空候補の生成・測定・選別]
    P --> Q
    Q --> S[Pareto frontier]
    S --> T[ブラインド比較レビュー]
    T --> U[近似と選好を分離した報告]
```

## 4. データモデル（ER図）

顔画像、ベクトル、人物テンプレート、学習fold、候補、レビューを追跡可能にする。生ベクトルと画像はローカルのみで、Gitへコミットしない。

```mermaid
erDiagram
    SOURCE_PROFILE ||--o{ SOURCE_IMAGE : 掲載する
    SOURCE_IMAGE ||--|| LOCAL_ASSET : 取得される
    LOCAL_ASSET ||--o| FACE_OBSERVATION : 検査される
    FACE_OBSERVATION ||--o| NEURAL_VECTOR : 合格時に生成
    FACE_OBSERVATION ||--o| GEOMETRY_VECTOR : 合格時に生成
    SOURCE_PROFILE ||--o{ DUPLICATE_CLUSTER : 人物内で持つ
    DUPLICATE_CLUSTER ||--o{ FACE_OBSERVATION : 構成する
    SOURCE_PROFILE ||--o| SUBJECT_TEMPLATE : 集約される
    SUBJECT_TEMPLATE }o--o{ EVALUATION_FOLD : 学習または評価に参加
    MODEL_RUN ||--o{ EVALUATION_FOLD : 実行する
    MODEL_RUN ||--|| MODEL_CONTRACT : 固定する
    MODEL_RUN ||--|| COMPONENT_MODEL : 生成する
    MODEL_RUN ||--o{ SCORE_RESULT : 採点する
    MODEL_RUN ||--o{ FICTIONAL_CANDIDATE : 計画する
    FICTIONAL_CANDIDATE ||--o{ SCORE_RESULT : 評価される
    REVIEW_SESSION ||--o{ PAIRWISE_RESPONSE : 含む
    FICTIONAL_CANDIDATE ||--o{ PAIRWISE_RESPONSE : 比較対象になる

    SOURCE_PROFILE {
        string subject_id PK
        string profile_url
        string source_host
        string adult_status
    }
    SOURCE_IMAGE {
        string image_id PK
        string subject_id FK
        string image_url
        string asset_class
    }
    LOCAL_ASSET {
        string image_id PK
        string local_path
        string sha256
        string retrieved_at
    }
    FACE_OBSERVATION {
        string observation_id PK
        string image_id FK
        int face_count
        float detector_confidence
        string landmarks_5
        string landmarks_106
        string quality_metrics
        boolean accepted
        string rejection_reason
    }
    NEURAL_VECTOR {
        string observation_id PK
        string contract_hash FK
        int dimension
        string local_vector_path
    }
    GEOMETRY_VECTOR {
        string observation_id PK
        string schema_version
        string local_vector_path
    }
    DUPLICATE_CLUSTER {
        string cluster_id PK
        string subject_id FK
        string representative_id FK
        string method_version
    }
    SUBJECT_TEMPLATE {
        string subject_id PK
        string model_run_id FK
        int accepted_clusters
        string neural_template_path
        string geometry_template_path
        string uncertainty
    }
    MODEL_CONTRACT {
        string contract_hash PK
        string source_manifest_sha256
        string clean_manifest_sha256
        string code_commit
        string backend_and_models
        string preprocessing_versions
    }
    MODEL_RUN {
        string model_run_id PK
        string contract_hash FK
        string algorithm
        string status
    }
    COMPONENT_MODEL {
        string model_run_id PK
        int component_count
        string center_path
        string components_path
        string bootstrap_stability
    }
    EVALUATION_FOLD {
        string fold_id PK
        string model_run_id FK
        string held_out_subject_id
        string fit_subject_ids_hash
    }
    SCORE_RESULT {
        string score_id PK
        string model_run_id FK
        string candidate_id FK
        float center_percentile
        float manifold_percentile
        float approximation_score
        boolean out_of_support
        string confidence_interval
    }
    FICTIONAL_CANDIDATE {
        string candidate_id PK
        string model_run_id FK
        string generator
        string generator_model
        int seed
        string prompt_hash
        string synthetic_label
    }
    REVIEW_SESSION {
        string review_session_id PK
        string panel_scope
        string randomization_seed
        string blinded_bundle_hash
    }
    PAIRWISE_RESPONSE {
        string response_id PK
        string review_session_id FK
        string left_candidate_id FK
        string right_candidate_id FK
        string question
        string choice
    }
```

## 5. 学習モデル構築シーケンス図

```mermaid
sequenceDiagram
    autonumber
    actor O as 操作者
    participant CLI as CLI
    participant DG as DataGate
    participant FE as FaceExtractor
    participant DV as DuplicateGate
    participant TM as TemplateModel
    participant CM as ComponentModel
    participant EV as LOSOEvaluator
    participant FS as LocalArtifactStore

    O->>CLI: audit-face-dataset(manifest, images)
    CLI->>DG: 出自・hash・asset classを検査
    DG->>FS: clean_manifest + rejected_manifest
    O->>CLI: build-face-observations(clean_manifest)
    loop 各画像
        CLI->>FE: 顔数・整列・品質・5点/106点
        FE->>FS: observation + neural/geometry vectors
    end
    O->>CLI: build-vector-model(observations)
    CLI->>DV: 人物内近重複をクラスタリング
    DV->>TM: 代表画像と品質重み
    TM->>FS: 人物別robust template
    CLI->>CM: 人物均等中心と安定成分をfit
    CM->>FS: model contract + components
    O->>CLI: evaluate-vector-model(model)
    loop 各held-out人物
        CLI->>EV: 学習人物だけで前処理から再fit
        EV->>EV: held-out templateをscore
        EV->>FS: fold manifest + metrics
    end
    EV->>EV: B0/B1/B2/C1と摂動を比較
    EV->>FS: promotion decision + report
    CLI-->>O: 採用またはB1へdefer
```

## 6. 1画像を採点するシーケンス図

```mermaid
sequenceDiagram
    autonumber
    actor O as 操作者
    participant CLI as score-face
    participant CT as ContractValidator
    participant FE as FaceExtractor
    participant SC as ApproximationScorer
    participant RP as Reporter

    O->>CLI: image + promoted_model
    CLI->>CT: model contractを検証
    alt 不一致または欠損
        CT-->>O: 採点せずエラー
    else 契約一致
        CLI->>FE: 単一顔・品質・整列を検査
        alt 顔ゲート不合格
            FE-->>O: scoreなし + rejection reason
        else 合格
            FE->>SC: unit neural vector + geometry + quality
            SC->>SC: center cosine百分位
            SC->>SC: tangent residual百分位
            SC->>SC: harmonic mean + support gate
            SC->>RP: score・内訳・CI・境界文
            RP-->>O: ローカル集合に対する近似結果
        end
    end
```

## 7. スイムレーン図

```mermaid
flowchart LR
    subgraph L1[操作者]
        A1[対象マニフェスト指定] --> A2[promotion結果確認]
        A2 --> A3[架空候補レビュー]
    end
    subgraph L2[データ品質]
        B1[出自・hash確認] --> B2[signature除外]
        B2 --> B3[単一顔ゲート]
        B3 --> B4[人物内近重複抑止]
    end
    subgraph L3[ベクトル化]
        C1[5点整列] --> C2[512D neural]
        C1 --> C3[106点 geometry]
    end
    subgraph L4[機械学習・統計]
        D1[人物別robust template] --> D2[spherical center]
        D2 --> D3[tangent components]
        D3 --> D4[LOSO校正・摂動]
        D4 --> D5[promotion gate]
    end
    subgraph L5[候補探索・評価]
        E1[生成・測定・選別] --> E2[Pareto frontier]
        E2 --> E3[ブラインド比較]
        E3 --> E4[近似と選好を分離報告]
    end
    subgraph L6[証跡保存]
        F1[immutable run manifest]
        F2[ローカルvector/image]
        F3[公開安全な集計report]
    end

    A1 --> B1
    B4 --> C1
    C2 --> D1
    C3 --> D1
    D5 --> A2
    D5 --> E1
    A3 --> E3
    B1 -.-> F1
    C2 -.-> F2
    C3 -.-> F2
    D4 -.-> F1
    E4 -.-> F3
```

## 8. 機械学習・校正フロー

```mermaid
flowchart TD
    A[有効人物集合 S] --> B[LOSO: hを1人選ぶ]
    B --> C[train = S - h]
    C --> D[trainだけで品質正規化]
    D --> E[trainだけで外れ値閾値を決定]
    E --> F[trainだけでcenter/componentsをfit]
    F --> G[held-out hを投影]
    G --> H[center agreement]
    G --> I[reconstruction residual]
    H --> J[全foldの経験分布 ECDF]
    I --> J
    J --> K[center percentile]
    J --> L[manifold percentile]
    K --> M[harmonic mean]
    L --> M
    F --> N[component support interval]
    N --> O{support内?}
    M --> P[Seju近似スコア]
    O -- いいえ --> Q[out_of_support=true]
    O -- はい --> R[out_of_support=false]
    P --> S1[bootstrap 95% CI]
    Q --> S1
    R --> S1
```

スコア式は次のとおりである。

```text
center_percentile = ECDF_LOSO(cosine(x, mu)) * 100
manifold_percentile = (1 - ECDF_LOSO(residual(x))) * 100
seju_approximation = harmonic_mean(center_percentile, manifold_percentile)
```

品質値は採点式に加算しない。品質とcomponent supportは、採点可能性および信頼度のゲートとして扱う。

## 9. promotion判定状態図

```mermaid
stateDiagram-v2
    [*] --> Diagnostic: clean dataset作成
    Diagnostic --> Blocked: 人物20未満または独立画像3未満
    Diagnostic --> BaselineReady: 最低証拠を満たす
    BaselineReady --> Evaluating: B0/B1/B2/C1 LOSO
    Evaluating --> BaselinePromoted: C1の改善下限が正でない
    Evaluating --> RobustPromoted: 全promotion gate合格
    Evaluating --> Blocked: 契約不一致・leakage・再現失敗
    BaselinePromoted --> CandidatePlanning
    RobustPromoted --> CandidatePlanning
    CandidatePlanning --> BlindReview: QA・support・score帯を通過
    BlindReview --> Reported: 選好と近似を別軸で集計
    Blocked --> Diagnostic: データまたは実装を修正
```

## 10. ベースラインと採用条件

| ID | 手法 | 人物均等 | 球面処理 | 品質考慮 | robust components |
|---|---|---:|---:|---:|---:|
| B0 | 画像重み付き算術平均 | いいえ | いいえ | いいえ | いいえ |
| B1 | 人物均等算術平均 | はい | いいえ | いいえ | いいえ |
| B2 | 人物均等球面平均 | はい | はい | いいえ | いいえ |
| C1 | 品質考慮球面幾何中央値＋安定成分 | はい | はい | はい | はい |

C1は、LOSO安定性の改善に対するbootstrap下限が正であり、worst-decile、摂動耐性、検出率が許容範囲内で、契約検証と再現実行が成功した場合だけ採用する。不合格時は単純なB1へ戻す。

## 11. 架空の3人目候補探索

512次元ベクトルを直接画像へ逆変換しない。生成器で複数候補を作り、同じpromoted modelで測定して残す `generate-measure-select` 方式を使う。

初期探索条件は次のとおり。

- Seju近似スコア70〜90。100を目標にしない。
- 安定component support内。
- 単一顔・画像品質・生成来歴の各ゲートに合格。
- 実在人物名、Seju公式を装う表現、保護属性や魅力ラベルをpromptに含めない。
- 架空候補同士のcomponent距離で多様性を確保。
- 最終判断は近似、support、QA、多様性、ブラインド選好のPareto frontier。

ブラインドレビューでは順序をランダム化し、seed、生成器、スコア、component目標を隠す。質問は `印象に残る`、`応援したい`、`SNSで見たい` を分け、`同程度` と棄権を許す。単独操作者の結果は個人選好であり、世間一般の人気とは表現しない。

## 12. CLIと成果物

| CLI | 入力 | 主な出力 | 失敗時の扱い |
|---|---|---|---|
| `audit-face-dataset` | source/download manifest、画像 | clean/rejected manifest、audit | fail closed |
| `build-face-observations` | clean manifest | observation、neural/geometry vector | 画像単位で理由記録 |
| `build-vector-model` | observations | subject templates、model contract | 証拠不足はdiagnostic |
| `build-vector-model`（分解を内包） | observations | templates、center、components | 最低人物数・独立portrait不足で停止 |
| `evaluate-vector-model` | model、folds | LOSO・摂動・promotion report | leakage/契約不一致で停止 |
| `score-face` | image、promoted model | score内訳・CI・support | ゲート不合格はscoreなし |
| `plan-fictional-candidates` | promoted model、探索設定 | candidate manifest | promotion前は拒否 |
| `review-fictional-candidates` | blind bundle、response | preference report | 近似情報混入で拒否 |

各runは、入力manifest hash、clean manifest hash、コードcommit、backend、detector、recognition model、各model hash、整列法、landmark schema、前処理、集約法、成分法、スコア定義versionを固定した契約を保存する。同じ次元でも契約が異なるベクトルは混合しない。

## 13. 検証マトリクス

| 検証層 | 必須確認 | 成功証拠 |
|---|---|---|
| 静的 | Ruff、構文コンパイル、型・契約キー | exit 0 |
| 単体 | gate、重複、集約、LOSO漏洩防止、score意味 | 全テストpass |
| 統合 | manifest→observation→template→component→score | 固定fixtureで再現一致 |
| 実データ | 259件再監査、signature除外、近重複分類 | 件数と理由付きmanifest |
| GPU | InsightFace CUDAで全有効画像 | provider、runtime、GPU memory |
| 統計 | 全LOSO fold、bootstrap、摂動 | CI・worst-decile・rank stability |
| promotion | B1対C1の事前条件 | 採用または明示defer |
| 候補 | synthetic provenance、score帯、support、QA、多様性 | candidate frontier |
| 人手 | blind順序、質問分離、比較数、panel scope | preference report |
| 安全 | raw vector/画像/人物別順位がGitにない | `git status`とartifact audit |

## 14. 完了条件

ベクトルモデル部分は、以下をすべて満たしたときだけ完了とする。

- 採用画像の出自とSHA-256が一致。
- signature、非顔、複数顔、近重複の余剰が理由付きで除外。
- 20人物以上かつ各人物3独立画像以上。
- neural vectorとgeometry vectorが別契約でversion管理。
- LOSOの学習人物と評価人物が完全分離。
- scoreの局所百分位という意味、CI、support外が表示される。
- B1またはC1のpromotion判断が実測に基づく。
- 本人照合、魅力度、属性、顔からの人気予測を出力しない。

架空候補部分は、promoted modelが存在し、候補がsyntheticとして追跡され、同じ条件でQA・採点され、候補間多様性を持ち、ブラインド選好が近似スコアと分離集計された場合だけ完了とする。

## 15. 実装順序

1. データゲートとclean manifest。
2. モデル契約、顔観測、106点幾何。
3. robust人物テンプレートと球面中心。
4. tangent-space component model。
5. LOSO校正Seju近似スコア。
6. B0/B1/B2/C1、摂動、CUDA実測とpromotion。
7. promoted modelに限定した架空候補planner。
8. 小規模生成・測定・Pareto選別。
9. ブラインド比較と分離レポート。

公開サイト、Discord取り込み、他事務所比較、SNS指標との相関、外部公開はこの仕様の完了条件に含めない。別の安全・権限・公開レビューを必要とする。
