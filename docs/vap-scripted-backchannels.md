# VAPターン境界と台本内相槌

この文書は、通常ターンをMaAI VAPで配置し、台本に最初から含まれる割り込み相槌をMaAIの
相槌専用`bc`モデルで配置する運用の正本です。ここでいう「相槌」は、新しい発話を後処理で
追加する機能ではありません。固定FTO方式は後方互換として残しています。

## 採用する構成

```bash
uv run zoom1-dialogue-tts input.txt -o out/dialogue.wav \
  --turn-timing vap-auto \
  --vap-python .venv-vap/bin/python \
  --backchannel-turn-timing vap \
  --backchannel-search-window-s 0.6 \
  --backchannels none
```

- 通常ターン: MaAI VAPのSHIFT/HOLDから間または重なりを決める
- 台本内の割り込み相槌: SHIFT/HOLDを使わず、`mode="bc"`の`p_bc`ピークへ配置する
- 新規の相槌音声: 追加しない（`--backchannels none`）
- 相槌配置後のVAP再解析: 行わない

`--backchannel-turn-timing`を省略した場合も`vap`が選ばれます。`--backchannel-turn-overlap-ms`
だけを指定した既存コマンドは後方互換のため`fixed`として扱います。

## 用語の区別

本ツールには、名前が似た2種類の相槌処理があります。

| 種類 | 入力台本に発話があるか | 配置方法 |
|---|---|---|
| 台本内相槌ターン | ある（例: `[S2]うん`） | `p_bc`ピーク、または後方互換の固定FTO |
| 後処理による追加相槌 | ない | `--backchannels stat`または`vap`で新しい相槌クリップを追加する |

今回採用する構成は前者だけを使用します。後者を確実に無効化するため、コマンドには
`--backchannels none`を指定できます。既定値も`none`であり、追加相槌は生成しません。

## 処理順序

1. FireRedTTS-2が台本を上から順に、各ターンをモノラル音声として1回生成する。
2. 割り込み相槌を解析対象から除き、16 kHzステレオの`*.vap_input.wav`を作る。
3. 10 Hzで`vap_jp_kyoto`と`vap_bc_jp`を実行する。相槌がなければ通常VAPだけを実行する。
4. 通常ターン境界は修正版VAPルールでFTOを予測する。
5. 相槌はテキストアンカー±0.6秒で`p_bc`が最大のフレームへ配置する。
6. 相槌音声を同じターン波形へ戻し、最終ステレオWAVを作る。

配置後の最終音声をMaAIへ再入力する反復処理は行いません。`p_bc`が自分の相槌音声を見て
答え合わせをしないよう、解析音声には相槌を含めません。

## 通常ターンのVAP判定

各チャンネルのフロアスコアは次の式です。

```text
floor_score = 0.65 * p_now + 0.35 * p_future
```

各境界では直前ターン末尾の800 ms前から100 ms前までを調べます。その窓内で次話者と前話者の
スコア差が最大になるフレームを求め、差が`0.15`以上ならSHIFT、未満ならHOLDです。

- SHIFT: 次話者が優位になった時刻から負のFTOを作る（VAP側の最大要求重なり500 ms）
- HOLD: 前話者の保持度から80〜1200 msの正のFTOを作る
- VAP予測がない境界: Zoom1統計へフォールバックする

## 相槌ターンの認識条件（MaAI bc方式）

短い相槌語に加えて、前後が同じ相手話者であり、直前発話が句点・疑問符・感嘆符で完結して
いないことを要求します。語彙だけで独立応答の「そうですね。」を相槌扱いするのを防ぎます。

## MaAI bcによる配置

台本上の直前実質ターン末尾をアンカーとし、既定では前後0.6秒の`p_bc`最大点を選びます。
相槌の個数と大まかな位置は台本が決め、モデルは近傍のどの100 msフレームへ置くかを決めます。
`p_bc`はチャネル非対称なので通常順と左右入替順を実行し、アンカー近傍の平均スコアが高い側を
採用します。配置に使わない `bc_2type` のreact/emo診断は標準経路では実行しません。

## 固定FTOと安全制約（後方互換）

1. 句読点と前後空白を除いた本文が次のいずれかに完全一致する。

   ```text
   うん / はい / ええ / ああ / へえ / うんうん / そうですね / なるほど
   ```

2. FireRedTTS-2が生成したターン音声の長さが1.2秒以下である。

部分一致は使いません。例えば「はい。きっと大丈夫です」は相槌ターンになりません。一方、
「そうですね。」は句点を除去すると一致するため、1.2秒以下なら相槌として扱われます。独立した
返答として扱いたい語がある場合は、`SHORT_BACKCHANNELS`の語彙を見直す必要があります。

`--backchannel-turn-overlap-ms 200`の場合、相槌ターンの要求値は常に次です。

```text
requested_offset_ms = -200.0
```

実適用値は通常の安全制約で制限されます。

```text
最大重なり = min(800 ms, 前後の短い方のターン長 * 0.5)
offset_ms = max(requested_offset_ms, -最大重なり)
```

例として、320 msの「うん」に対する最大重なりは160 msなので、要求`-200 ms`に対する実適用は
`-160 ms`です。400 ms以上の相槌は、直前ターンも十分長ければ`-200 ms`がそのまま適用されます。

## Manifestと診断ファイル

相槌特別扱いの境界は最終manifestで次のように記録されます。

```json
{
  "turn_index": 18,
  "requested_offset_ms": -200.0,
  "offset_ms": -160.0,
  "event": "backchannel",
  "source": "backchannel-fixed",
  "score": 1.0
}
```

`score: 1.0`は固定ルールを示す値であり、MaAIの確率ではありません。通常境界では`event`が
`shift`または`hold`になります。

`vap-auto`は出力WAVと同じディレクトリへ次も保存します。

- `*.vap_input.wav`: VAP調整前の重なりなし音声
- `*.vap_input.manifest.json`: VAP解析用ターン位置
- `*.vap_trace.json`: 100 msごとの`p_now`/`p_future`
- `*.vap_turns.json`: VAPまたは固定ルールが要求した境界FTO
- `*.manifest.json`: 安全制約適用後の最終FTOと成果物への参照

## 参照音声付き・複数ファイルの例

```bash
HF_HOME=/path/to/huggingface \
zoom1-dialogue-tts \
  --batch inputs/dialogue1.txt out/dialogue1.wav \
  --batch inputs/dialogue2.txt out/dialogue2.wav \
  --prompt-s1 references/system.wav "参照音声の正確な書き起こし" \
  --turn-timing vap-auto \
  --vap-python .venv-vap/bin/python \
  --vap-device cpu \
  --backchannel-turn-timing vap \
  --backchannel-search-window-s 0.6 \
  --backchannels none \
  --seed 1
```

`--batch`ではFireRedTTS-2を1回だけロードし、複数ファイルを順番に処理します。MaAI解析は各出力の
`vap_input.wav`に対して個別に実行されます。
