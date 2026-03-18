# 数据说明

本文档是 `data/README.md` 的中文扩展版，放在 `data/` 目录中，目的是把数据准备过程说清楚到可以直接照着落地。

它重点回答五个问题：

1. 每个数据集应该从哪里下载。
2. 下载后的原始文件建议放在哪个目录。
3. 当前项目实际会读取哪些字段。
4. 原始数据怎样被转换成 `original.json`、`documents.json` 和 `kg.json`。
5. 使用这些数据集时最常见的问题是什么。

## 一、当前 `data` 目录里有什么

当前目录里已经有以下内容：

- `demo_original.json`：手写的最小示例 anchor 数据。
- `demo_documents.json`：手写的最小示例文档集合。
- `demo_kg.json`：手写的最小示例知识图谱。
- `raw_samples/`：四个 benchmark 的最小原始格式样例。
- `README.md`：英文详细说明。
- `README.zh-CN.md`：当前这份中文说明。

建议你后续按下面的目录组织真实实验数据：

```text
data/
  README.md
  README.zh-CN.md
  raw/
    webqsp/
      train.json
      test.json
    hotpotqa/
      hotpot_train_v1.1.json
      hotpot_dev_distractor_v1.json
    nq/
      train.jsonl
      dev.jsonl
    triviaqa/
      wikipedia-train.json
      wikipedia-dev.json
  kg/
    freebase_subset.json
    wikidata_subset.json
  prepared/
    webqsp/
    hotpotqa/
    nq/
    triviaqa/
  scripts/
    prepare_dataset.ps1
    run_dataset.ps1
```

## 二、S-Gens 在当前工程里真正需要什么数据

无论原始 benchmark 长什么样，最后都会先统一成三份文件：

### 1. `original.json`

这是原始监督锚点集合，每条样本至少包含：

- `id`：样本编号。
- `query`：查询或问题文本。
- `positive_doc_id`：正样本文档编号，如果能从原始数据中识别出来。
- `core_entities`：核心实体列表，后续用于在 KG 中查找推理路径。

### 2. `documents.json`

这是文档集合，每条文档包含：

- `id`：文档编号。
- `text`：文档正文。

### 3. `kg.json`

这是外部知识图谱。注意，它不是从这些 benchmark 里自动长出来的，而是你额外准备的。

当前项目接受两种格式：

```json
{
  "triples": [
    ["Christopher Nolan", "directed", "Inception"],
    ["Inception", "stars", "Leonardo DiCaprio"]
  ]
}
```

或者：

```json
[
  {"head": "Christopher Nolan", "relation": "directed", "tail": "Inception"},
  {"head": "Inception", "relation": "stars", "tail": "Leonardo DiCaprio"}
]
```

## 三、为什么必须单独准备 KG

S-Gens 不是只靠问答数据本身工作，而是依赖外部知识图谱来做：

- 多跳路径抽取
- 路径引导的 query 生成
- 结构一致正样本筛选
- 结构性 hard negative 构造

所以：

- benchmark 数据负责提供 query、answer、context 或搜索页
- KG 负责提供实体之间的显式关系路径

如果没有 KG，或者 KG 太稀疏，就会出现：

- 找不到推理路径
- 生成不出 synthetic positives
- 生成不出结构 hard negatives

## 四、四个数据集分别怎么下、怎么处理

## 1. WebQSP

### 数据集性质

WebQSP 是面向知识库问答的数据集，问题通常隐含一条 Freebase 风格的关系链。

它更像“知识图谱问答数据”，不太像“自带 passage 的检索数据”。

### 建议下载方式

建议下载 WebQSP 的 JSON 版本，通常常见字段包括：

- `Question`
- `Parses`
- `TopicEntityName`
- `InferentialChain`
- `Answers`

你下载后建议放到：

```text
data/raw/webqsp/train.json
```

### 当前项目会读取哪些字段

当前实现会优先读取：

- `Question` / `question` / `RawQuestion`
- `TopicEntityName` 或 `Parses[0].TopicEntityName`
- `Parses[].Answers[]`
- `SupportingText`，如果你额外补了这个字段

### 进入当前工程后的处理方式

每条 WebQSP 样本会被转成：

1. `query`
   直接取问题文本。
2. `core_entities`
   由 topic entity 和前几个 answer entity 组成。
3. `positive_doc_id`
   当前实现会人为生成一个形如 `webqsp-doc-{idx}` 的文档 id。
4. `documents.json` 中的正文
   如果原始数据里有 `SupportingText`，就直接用它。
   如果没有，就退化为一段拼接文本：问题 + topic entity + answer list + inferential chain。

### 你要特别注意什么

WebQSP 最大的问题是它通常不自带真正的 passage 语料。

因此它在当前工程中有两种用法：

1. 轻量调通流程
   只用原始 WebQSP JSON，让系统能跑通。
2. 真正做检索复现
   除了 WebQSP 问题，还要给它补一个真实文本语料库，或者给每条样本补 `SupportingText`。

如果你只给 WebQSP 原始问答文件，不补文本语料，那么生成出的 `documents.json` 会更像“弱监督伪文档”，不是严格意义上的检索证据。

### 更适合配什么 KG

WebQSP 最自然的是配 Freebase 风格子图。

## 2. HotpotQA

### 数据集性质

HotpotQA 是典型的多跳问答数据集，自带 supporting facts 和 context 文档。

它是当前项目里最适合直接接入的 benchmark。

### 建议下载方式

建议下载官方发布版本，例如：

- `hotpot_train_v1.1.json`
- `hotpot_dev_distractor_v1.json`

放到：

```text
data/raw/hotpotqa/hotpot_train_v1.1.json
```

### 当前项目会读取哪些字段

当前实现会读取：

- `question`
- `answer`
- `supporting_facts`
- `context`

### 进入当前工程后的处理方式

每条 HotpotQA 样本会被处理成：

1. `query`
   直接取 `question`。
2. `documents.json`
   `context` 中每个 `[title, sentences]` 都会变成一篇文档。
   文本会被拼成：`title + 句子列表`。
3. `positive_doc_id`
   当前实现选取第一个 title 出现在 `supporting_facts` 中的 context 作为正样本文档。
4. `core_entities`
   由 `supporting_facts` 中的 title，再加上 `answer` 组成。

### 你要特别注意什么

HotpotQA 的真实监督其实往往涉及多篇 supporting documents，但当前实现先简化成了“单正样本入口”。

所以它现在的作用是：

- 非常适合快速验证整条 S-Gens 逻辑
- 但还不是完整的 multi-positive retrieval 训练实现

### 更适合配什么 KG

HotpotQA 更适合配 Wikidata 子图，或者 Wikipedia 实体关系抽取图。

## 3. NQ

### 数据集性质

Natural Questions 是开放域问答数据集。

很多研究不会直接用最原始的 Google 标注格式，而是用已经整理好的 open-domain 版本。

### 建议下载方式

建议优先使用已经带 passages 或 contexts 的 open-domain NQ 版本。

当前项目更适合这种格式：

- `question` 或 `query`
- `contexts` 或 `passages`

放到：

```text
data/raw/nq/train.jsonl
```

### 当前项目会读取哪些字段

当前实现会读取：

- `question` / `query`
- `contexts` / `passages`

对于每个 context/passages，会继续读取：

- `text`
- `passage_text`
- `context`
- `title`
- `is_positive`
- `has_answer`

### 进入当前工程后的处理方式

每条 NQ 样本会被处理成：

1. `query`
   直接取问题文本。
2. `documents.json`
   每个 context 都转成一篇文档。
3. `positive_doc_id`
   优先选取显式带 `is_positive` 或 `has_answer` 标记的 context。
   如果没有，就退化为第一个 context。
4. `core_entities`
   当前仅用简单启发式抽取，例如问题中的大写词和少量元字段。

### 你要特别注意什么

NQ 当前的薄弱点不是 passage，而是实体链接。

也就是说：

- passage 侧通常够用
- 但 `core_entities` 质量还不高
- 如果以后你想更接近论文设定，最好补一个实体链接器

### 更适合配什么 KG

NQ 比较适合配 Wikidata 或 Wikipedia 实体图。

## 4. TriviaQA

### 数据集性质

TriviaQA 是大规模事实型问答数据集，通常带有搜索结果页或实体页。

它比 WebQSP 更适合作为检索任务入口，因为它更容易拿到外部证据文本。

### 建议下载方式

建议使用带证据页的版本，例如 Wikipedia-backed 或 web-backed TriviaQA。

原始字段常见为：

- `Question`
- `Answer`
- `SearchResults`

或者：

- `question`
- `EntityPages`

放到：

```text
data/raw/triviaqa/wikipedia-train.json
```

### 当前项目会读取哪些字段

当前实现会读取：

- `Question` / `question`
- `Answer.Value`
- `Answer.Aliases`
- `SearchResults` / `EntityPages`

对于每个 evidence page，会读取：

- `Title`
- `Snippet`
- `text`

### 进入当前工程后的处理方式

每条 TriviaQA 样本会被处理成：

1. `query`
   问题文本。
2. `documents.json`
   每个搜索结果页都变成一篇文档。
3. `positive_doc_id`
   当前用简单规则选取第一个包含 answer alias 的页面。
4. `core_entities`
   由问题中的命名词和前几个 answer alias 组成。

### 你要特别注意什么

TriviaQA 当前的主要近似在于正样本选择。

因为现在只是用答案字符串命中来判正样本，所以：

- 调通流程没问题
- 但严格实验时应尽量换成更可靠的 relevance 标注或更强的 passage 过滤逻辑

### 更适合配什么 KG

TriviaQA 比较适合配 Wikidata 或 Wikipedia 实体关系图。

## 五、KG 应该怎么准备

### 当前项目支持什么格式

KG 文件既可以写成：

```json
{
  "triples": [
    ["head", "relation", "tail"]
  ]
}
```

也可以写成：

```json
[
  {"head": "head", "relation": "relation", "tail": "tail"}
]
```

同时兼容以下字段别名：

- `subject` 或 `s` 作为 head
- `predicate` 或 `p` 作为 relation
- `object` 或 `o` 作为 tail

### 推荐准备方式

- WebQSP：优先 Freebase 子图。
- HotpotQA：优先 Wikidata 子图或 Wikipedia 实体关系图。
- NQ：优先 Wikidata / Wikipedia 图。
- TriviaQA：优先 Wikidata / Wikipedia 图。

### 质量要求

你的 KG 至少要满足一个条件：

对于 `core_entities` 中的实体，图里要有足够多的连接关系，能支撑 2 到 4 跳的路径搜索。

如果不满足，就会导致：

- 路径搜索失败
- 没有正样本
- 没有 hard negatives

## 六、推荐执行顺序

### 方案 A：只做标准化

```bash
python -m sgens.cli prepare-dataset \
  --dataset hotpotqa \
  --raw data/raw/hotpotqa/hotpot_train_v1.1.json \
  --kg data/kg/wikidata_subset.json \
  --output-dir data/prepared/hotpotqa
```

执行后你会得到：

- `data/prepared/hotpotqa/original.json`
- `data/prepared/hotpotqa/documents.json`
- `data/prepared/hotpotqa/kg.json`
- `data/prepared/hotpotqa/metadata.json`

### 方案 B：标准化后直接跑 S-Gens

```bash
python -m sgens.cli run-dataset \
  --dataset hotpotqa \
  --raw data/raw/hotpotqa/hotpot_train_v1.1.json \
  --kg data/kg/wikidata_subset.json \
  --output-dir outputs/hotpotqa
```

它会先在 `outputs/hotpotqa/normalized/` 里写入标准化结果，再在 `outputs/hotpotqa/sgens/` 里写入生成结果。

## 七、最常见的问题

### 1. 为什么没有生成 synthetic positives

通常有三个原因：

- `core_entities` 和 KG 里的实体命名对不上。
- KG 中找不到从源实体到目标实体的路径。
- passage 里没有覆盖足够多的路径实体，过不了 path coverage 阈值。

### 2. 为什么 WebQSP 效果差

这通常不是代码错，而是因为 WebQSP 原始数据本身缺少真正的 passage 语料。

### 3. 为什么 NQ 或 TriviaQA 的正样本看起来不稳定

因为当前实现为了先把流程复现出来，使用的是轻量规则：

- 优先用 `has_answer` / `is_positive`
- 否则用答案字符串命中

这不是最终版实验配置，而是当前复现阶段的工程折中。

### 4. 为什么 JSON 明明没问题却读取失败

如果文件带 BOM，普通 `utf-8` 读取会报错。当前项目已经改成 `utf-8-sig`，所以这种情况现在应该可以正常处理。

## 八、建议你先从哪个数据集开始

如果你的目标是：

- 先验证整条流程能跑通
- 尽快看到 triplet 结果

那就先用 HotpotQA。

如果你的目标是：

- 更贴近论文里“知识图谱驱动的推理路径”设定

那 WebQSP + Freebase 子图更合适，但它对数据工程要求更高。
