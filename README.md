# Zoom1 Dialogue TTS — custom inference workflow

[Demo & audio samples](https://llm-jp.github.io/zoom1-dialogue-tts/) | [Hugging Face model](https://huggingface.co/llm-jp/zoom1-dialogue-tts)

FireRedTTS-2をLLM-jp Zoom1の日本語対話でfine-tuneし、2話者の自然なステレオ対話音声を
生成する推論ツールです。左チャンネルがS1、右チャンネルがS2です。FireRedTTS-2が過去の
両話者のテキスト・音声トークンをインターリーブした文脈から各ターンを生成し、VAPまたは
Zoom1統計に基づく時間編集で、話者交代時の間・重なりと聞き手の相槌を付与します。

- Model: [llm-jp/zoom1-dialogue-tts](https://huggingface.co/llm-jp/zoom1-dialogue-tts)
- Base implementation: [FireRedTeam/FireRedTTS2](https://github.com/FireRedTeam/FireRedTTS2)
- Output: 24 kHz, 2-channel WAV (left=S1, right=S2)

## このリポジトリについて

このリポジトリは、提供を受けたZoom1 Dialogue TTSの推論ページ／実装を、対話音声合成の
比較実験向けにカスタマイズして保存したものです。学習済みモデルそのものを再配布する
リポジトリではありません。主な変更点は次のとおりです。

- 再現可能な生成用の`--seed`と`--temperature`（既定値は1と0.8）
- S1/S2それぞれに限定した参照音声と正確な文字起こし、および`first-turn` prompt scope
- MaAI VAPによる通常対話ターンの間・重なりの配置
- MaAIの`bc`モデルによる、台本内の割り込み相槌専用の配置
- 相槌を除いた解析音声を使うことによる、相槌タイミング推定時の音声リーク防止

元実装、同梱コードの由来、ライセンス上の扱いは[NOTICE](NOTICE)を参照してください。

## 標準の合成方式

標準設定は、`drop`・seed 1・temperature 0.8・top-k 20・BF16でターン音声を生成し、
通常ターンを `maai-kyoto/vap_jp_kyoto`、台本内相槌を `maai-kyoto/vap_bc_jp` で配置します。
モデルrevisionはコードで固定しています。台本にない相槌は追加しません。
同梱の `references/turn000_S1.wav` と書き起こしをS1の最初のターンだけに適用します。
別の参照は `--prompt-s1`、参照を使わない場合は `--no-prompt-s1` で指定できます。

生成履歴のリサンプルと音声トークンキャッシュはGPU上で扱い、対話生成後にCPUへ移します。
ターンWAVとmanifestはVAP配置前に保存し、同じ入力・設定で再実行すると再利用します。
CPU/GPUやPBSの使い方は[合成方式とバッチ実行](docs/dataset-synthesis.md)を参照してください。

## 必要なモデルとアクセス権

GitHubから取得できるのは推論コードだけです。実行時には次のモデルが必要です。

| 用途 | モデル／パッケージ | 読み込み方法 |
|---|---|---|
| 音声生成 | `llm-jp/zoom1-dialogue-tts` | Hugging Faceから自動取得。現在はprivateのためアクセス権が必要 |
| codec/tokenizer | `FireRedTeam/FireRedTTS2` | Hugging Faceから必要ファイルだけ自動取得 |
| 通常ターン配置 | [`maai-kyoto/vap_jp_kyoto`](https://huggingface.co/maai-kyoto/vap_jp_kyoto)（MIT） | 別の`.venv-vap`環境で固定revisionを読み込む |
| 台本内相槌配置 | [`maai-kyoto/vap_bc_jp`](https://huggingface.co/maai-kyoto/vap_bc_jp)（MIT） | 同じ`.venv-vap`環境で固定revisionを読み込む |

モデル重み、Hugging Face token、ユーザー固有の参照音声はGitへcommitしないでください。
参照音声には、利用・再配布の許諾を得た音声だけを使用してください。

> [!IMPORTANT]
> 現在、モデルリポジトリはprivateです。初回実行前にアクセス権のあるHugging Face
> アカウントで `hf auth login` を実行してください。公開時にはこの注意を更新します。

アクセス権のある共同研究者向けに、Zoom1収録話者の許諾範囲内で実音声をvoice-clone
プロンプトとして用いた比較サンプルを`private_samples/`へ収録しています。このサンプルは
公開GitHub Pagesには含まれません。利用条件は同ディレクトリのREADMEを確認してください。

## セットアップ

CUDA GPUを搭載したLinux環境を想定しています。`uv`を使うとPythonとCUDA版PyTorchを
含む環境を再現できます。

```bash
git clone https://github.com/Kitamura-Kx/zoom1-dialogue-tts-mycustom.git
cd zoom1-dialogue-tts-mycustom
uv sync --extra test
uv run hf auth login  # モデルがprivateの間だけ
```

初回推論時に、Zoom1 fine-tuneと公式FireRedTTS-2のcodec/tokenizerをHugging Faceから
自動取得します。`drop`と`keep`のうち、指定した一方のチェックポイントだけを取得します。
モデルと公式baseは検証済みrevisionに固定され、不要なbase LLM重みは取得しません。

既定モデルへアクセスできない場合は、同じFireRedTTS-2レイアウトのチェックポイントを
`--model-id OWNER/REPOSITORY`で指定するか、`ZOOM1_TTS_MODEL_ID`環境変数で設定します。
ローカルの参照音声は`--prompt-s1`／`--prompt-s2`で実行時に渡し、リポジトリ内へ置く必要は
ありません。

## クイックスタート

入力は1行1ターンです。

```text
[S1]お疲れさまです。この前の連休、温泉に行ってきたんですよ。
[S2]いいですね。どのあたりに行ったんですか。
```

```bash
# 標準: Kyoto VAPの通常ターン配置と台本内相槌配置
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/dialogue.wav
```

生成物:

- `out/dialogue.wav`: VAP配置した通常ターンと台本内相槌を含むステレオ音声
- `out/dialogue.manifest.json`: 各ターンと境界の開始時刻、FTO、相槌、VAP診断ファイル
- `out/dialogue_turns/`: 後処理前のターン別音声

## 声を指定する

収録許諾のある参照音声と、その正確な書き起こしを話者ごとに指定できます。3〜10秒程度の
単一話者・無雑音音声を推奨します。

```bash
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/dialogue.wav \
  --prompt-s1 voice_s1.wav "参照音声の書き起こし" \
  --prompt-s2 voice_s2.wav "参照音声の書き起こし"
```

参照音声を省略した場合は、同梱のS1参照音声を使用します。`--no-prompt-s1` で参照なしにできます。
標準設定は台本内相槌の元音声を配置します。追加相槌の再合成は `--backchannels stat/vap` を明示した場合だけです。

参照音声が相手話者の生成へ直接影響するのを避けたい場合は、参照を指定話者の最初のターン
だけに適用できます。その後は、最初に生成したターンを含む通常の対話履歴だけを使用します。

```bash
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/first-turn-prompt.wav \
  --prompt-s1 voice_s1.wav "参照音声の書き起こし" \
  --prompt-scope first-turn
```

既定は`--prompt-scope first-turn`です。参照音声を全ターンの文脈へ含める従来方式が
必要な場合だけ、`--prompt-scope all`を明示してください。

生成の既定値は、比較評価で安定していた`--temperature 0.8`、`--seed 1`です。
したがって通常はこれらを省略でき、変更したい場合だけ明示します。

### 複数ファイルを一度のモデルロードで生成する

`--batch INPUT OUTPUT`を繰り返すと、モデルを一度だけロードして複数の対話を順番に生成
できます。参照音声や生成設定は全入力で共通です。

```bash
uv run zoom1-dialogue-tts \
  --batch inputs/dialogue1.txt out/dialogue1.wav \
  --batch inputs/dialogue2.txt out/dialogue2.wav \
  --prompt-s1 voice_s1.wav "参照音声の書き起こし"
```

## 相槌タイミング

追加相槌を明示的に有効にする`stat`モードは、Zoom1の相槌頻度（約3.1回/分）に基づいて長い発話へ相槌を配置します。

```bash
# 追加相槌を有効化
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/stat.wav \
  --backchannels stat --bc-per-minute 3.1

# 相槌なし
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/no_bc.wav --backchannels none
```

VAPで検出した時刻を使う場合は、`[{"time": 2.2, "listener_channel": 1, "score": 0.65}]`
形式のJSONを渡します。`time`は後処理前の音声上の秒、チャンネルは0=S1、1=S2です。

```bash
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/vap.wav \
  --backchannels vap --vap-json vap_points.json
```

VAP検出器はMaAIとPyAudioの環境制約があるため、本パッケージの標準依存には含めていません。
標準方式にはVAP環境が必要です。VAPを使わない実験では `--turn-timing stat` または `none` を指定できます。

## ターン交替タイミング

通常ターンの間・重なりもVAPの`p_now`/`p_future`で文脈依存化できます。VAPが次話者への
SHIFTをターン末尾より前に予測した境界は重ね、現在話者の継続を予測した境界には間を置きます。
予測がない境界だけZoom1統計へfallbackし、短いターンへの過剰な食い込みは自動で制限します。

MaAIはFireRedTTS-2と依存関係が異なるため、初回だけ別環境を作成します。PortAudioを導入
できない計算ノードでも、本ツールがWAV解析時にPyAudioを自動スタブ化するため利用できます。
`--turn-timing stat`または`none`だけを使う場合、このMaAI環境は不要です。

```bash
uv venv --python 3.12 .venv-vap
uv pip install --python .venv-vap/bin/python --no-deps maai==0.2.13
uv pip install --python .venv-vap/bin/python \
  torch torchaudio numpy soundfile librosa einops rich matplotlib scipy \
  transformers==5.5.3 huggingface-hub pygame
```

```bash
# 推奨: FireRed生成、VAP解析、FTO適用を1コマンドで実行
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/vap_dialogue.wav \
  --turn-timing vap-auto --vap-python .venv-vap/bin/python --seed 1
```

FireRedTTS-2による各ターンの生成は1回だけです。生成後にターン分離音声をMaAIへ渡し、
予測FTOを同じ音声へ適用します。`out/vap_dialogue.vap_input.wav`、VAP trace、境界JSONも
診断用に保存され、最終manifestの`vap_artifacts`から参照できます。

### 台本内相槌をMaAI相槌モデルで配置する（現在の推奨構成）

`vap-auto`では、台本内の割り込み相槌を通常のSHIFT/HOLD判定から除外し、MaAIの相槌専用
`bc`モデルが出す`p_bc`のピークへ配置します。解析音声から相槌自体を除くため、モデルが
すでに鳴った相槌を見てしまうリークも避けます。後処理で新しい相槌を追加しない運用では、
追加相槌を止める`--backchannels none`も指定します。

```bash
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/vap_dialogue.wav \
  --turn-timing vap-auto --vap-python .venv-vap/bin/python \
  --backchannel-turn-timing vap \
  --backchannel-search-window-s 0.6 \
  --backchannels none
```

相槌候補は短い相槌語であり、かつ「前後が同じ相手話者」「直前発話が句点等で完結していない」
ターンです。独立応答としての「そうですね。」は通常ターンのまま残します。台本上の直前位置を
アンカーとし、その前後`0.6`秒で`p_bc`最大の100 msフレームを選びます。`p_bc`はチャネル
非対称なので、左右を入れ替えた解析も行い、アンカー近傍の平均スコアが高い順序を自動採用します。
配置に使わない `bc_2type` 推論は標準経路では実行しません。任意の診断には `bc_cli --modes bc_2type` を使えます。

従来の固定FTOが必要な場合は、後方互換モードを明示できます。

```bash
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/fixed_bc.wav \
  --turn-timing vap-auto --vap-python .venv-vap/bin/python \
  --backchannel-turn-timing fixed --backchannel-turn-overlap-ms 200 \
  --backchannels none
```

台本内相槌と後処理による追加相槌の違い、処理順、SHIFT/HOLD式、manifestの読み方を含む正確な
仕様は[「VAPターン境界と台本内相槌」](docs/vap-scripted-backchannels.md)を参照してください。

`vap_turns.json`は次の形式です。負値が重なり、正値が間です。

```json
[
  {"turn_index": 1, "offset_ms": -320.0, "score": 0.61, "source": "vap-shift"},
  {"turn_index": 2, "offset_ms": 180.0, "score": 0.24, "source": "vap-shift"}
]
```

MaAIは標準依存には含めていません。保存したVAP traceを`--trace-json`で再利用すれば、
モデルを再実行せず閾値を比較できます。従来の統計FTOは`--turn-timing stat`、重なりなしは
`--turn-timing none`です。

閾値を比較するときは、従来の手動経路も利用できます。

```bash
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/base.wav \
  --turn-timing none --backchannels none --seed 1
.venv-vap/bin/python tools/vap_turn_timing.py \
  out/base.wav out/base.manifest.json out/vap_turns.json --save-trace out/vap_trace.json
uv run zoom1-dialogue-tts examples/dialogue.txt -o out/vap_manual.wav \
  --turn-timing vap --turn-vap-json out/vap_turns.json --seed 1
```

## モデルvariant

```bash
# 推奨: 被せ相槌を除いたZoom1でfine-tune。発話が比較的明瞭
uv run zoom1-dialogue-tts examples/dialogue.txt --variant drop -o out/drop.wav

# 学習時に短い相槌ターンを保持
uv run zoom1-dialogue-tts examples/dialogue.txt --variant keep -o out/keep.wav
```

## 入力形式

`.txt`、`.json`、`.jsonl`に対応します。JSON/JSONLではA/BをS1/S2へ変換します。

```json
{
  "turns": [
    {"speaker": "S1", "text": "こんにちは。"},
    {"speaker": "S2", "text": "こんにちは。"}
  ]
}
```

## 仕組み

1. 過去のS1/S2発話をインターリーブした文脈から、FireRedTTS-2が次のターンをモノラル生成
2. ターンをS1=左、S2=右へ分離
3. VAP SHIFT予測、またはZoom1由来のFTO分布から、話者交代ごとの間・重なりを決定
4. 統計配置またはVAP指定時刻に、聞き手と同じ声の相槌を挿入
5. ステレオWAVと再現用manifestを保存

FireRedTTS-2は両話者の履歴を対話文脈として利用しますが、2チャネル波形を同時生成する
full-duplexモデルではありません。各ターンを順番に生成し、左右チャンネルと同時発話を後段で
構成します。この構成により、発話生成とターンタイミングを独立に比較・再実行できます。

### KABURI-TTSとの違い

[KABURI-TTS](https://github.com/llm-jp/kaburi-tts)は、予測した2話者の発話活動を条件として、
固定30秒canvas上のA/B音声latentをtwo-stream rectified-flow音響モデルで生成します。本ツールは
FireRedTTS-2のAR対話文脈からターンを逐次生成し、VAPでターン交替と相槌の時間構造を付与します。

| | KABURI-TTS | Zoom1 Dialogue TTS |
|---|---|---|
| 音響生成 | 2-stream、30秒の2チャネルlatent | AR、文脈付きターン逐次生成 |
| ターン交替 | テキスト条件のタイミング予測器 | 音声条件のVAP SHIFT/HOLD |
| 重なり | 発話活動を音響モデルへ条件付け | 個別生成したターンを時間編集 |
| 長さ | 現実装は30秒固定 | 1ターン30秒、全体はコンテキスト上限まで |

両者は同じ目的に対する異なる構成であり、本リポジトリはFireRedTTS-2の対話履歴モデリングと
VAPによる韻律依存タイミングを分離して評価できる点を主な位置づけとします。

FireRedTTS-2の学習コード、デモ、Docker資産は同梱しません。公式リポジトリの固定commitから
推論に必要な`fireredtts2`パッケージ（約144 KB）だけを収録し、NOTICEに由来と変更点を
記載しています。それ以外は推論ラッパーとZoom1風タイミング処理です。

## 制限事項

- FireRedTTS-2は逐次的にターンを生成します。重なりと相槌は生成後の時間編集であり、
  full-duplexモデルのように両話者を同時生成しているわけではありません。
- 「うん」「はい」以外のおうむ返しや意味のある割り込み内容は、入力台本に含める必要があります。
- VAP SHIFTを使わない既定設定では、通常ターンの食い込みはZoom1統計から決まります。
- `vap-auto`は1コマンドですが、内部ではFireRedTTS-2生成後に別環境のMaAIを実行します。
- 参照声には、本人の同意と利用許諾がある音声だけを使用してください。
- 1ターンの既定生成上限は30秒です。対話全体もLLM系列長`3100`トークンに制約され、過去の
  音声トークンが累積するため、上限に達した場合は最古の生成履歴から外すsliding contextを使います。
  外した履歴数はターンmanifestに記録します。

## 開発

```bash
uv run pytest
```

テストはモデルをダウンロードせず、台本解析、モデル構成、タイミング処理を検証します。

## ライセンス

本リポジトリのコードはApache License 2.0です。同梱するFireRedTTS-2の実装もApache License 2.0です。
標準の時間配置に使う以下の2モデルは、それぞれの固定revisionのモデルカードでMITと表記されています。

| 用途 | モデル | ライセンス表記の出典 |
| --- | --- | --- |
| 通常ターンの間・重なり | `maai-kyoto/vap_jp_kyoto` | [MIT（固定revisionのモデルカード）](https://huggingface.co/maai-kyoto/vap_jp_kyoto/blob/fe24ac60d8fcc80463edde97ed90e3ceca5e5b88/README.md) |
| 台本内相槌の時間配置 | `maai-kyoto/vap_bc_jp` | [MIT（固定revisionのモデルカード）](https://huggingface.co/maai-kyoto/vap_bc_jp/blob/309d7936f5a929870e3e02a563729b7dc8b78a6e/README.md) |

商用の下流モデル利用を目的とした再作成に合わせ、通常ターン配置をMIT表記の
`vap_jp_kyoto`へ変更しています。生成済み音声や学習済みモデル全体の利用条件は、
TTS重み・参照音声・入力台本などの条件にも従います。コードのApache License 2.0や
VAPモデルのMIT表記が、それらの利用権を一括して付与するものではありません。
各モデルのHugging Face model cardと音源・台本の利用条件を確認してください。

## 謝辞

- FireRedTeam, FireRedTTS-2
- LLM-jp Zoom1 dataset and Dialogue Working Group
- MaAI/VAP backchannel model (optional timing detector)
