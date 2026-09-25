<p align="center"><img src="assets/kevane-logo.png" alt="KevANE 标志" width="320"></p>

# KevANE

[English](README.md) · [模型文件与模型卡](https://huggingface.co/flylcw/KevANE-0.6B) · [许可证](LICENSE)

KevANE 让 **Kev 0.6B 判断模型**在 Apple Silicon 本机运行。项目将 Qwen3 主干转换为 Core ML，并提供兼容 TypeSafe 的 `POST /v1/systemone` 接口，可供 [Jev Jarvis](https://github.com/jev-chat/jev-chat-jarvis-mac) 等客户端调用。它处理选择、评分和是非判断，不负责生成回复文字。

## 为什么用 Kev？为什么用 ANE？

- **Kev** 提供 Jarvis 使用的 System One 判断格式：输入上下文和问题，输出各选项的概率。KevANE 固定使用[已验证版本](src/kev/UPSTREAM.txt)的编码器、接口定义和 Pointer Head。
- **ANE** 是 Apple Silicon 上 Core ML 可用的计算设备。KevANE 以 `CPU_AND_NE` 加载主干模型，排除 GPU；执行计划显示许多操作首选 ANE，但这不等于逐次请求的硬件归因。分词和 Pointer Head 仍由 CPU 处理。

## 安装

需要 Apple Silicon Mac、[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)，以及供 Core ML 模型和本机编译缓存使用的数 GB 空间。只需在首次安装时进入源码目录：

```bash
git clone https://github.com/LCW0NJUPT/KevANE.git
cd KevANE
bash scripts/install.sh
```

安装脚本会明确选用 `kevane-runtime` Conda 环境，不受当前激活环境影响；必要时创建该环境，下载前先核对模型仓库文件，并创建 `~/.local/bin/kev-ane` 命令。服务直接从源码目录导入代码，安装脚本不会执行 `pip install -e`，也**不会自动启动服务**。如果终端的 `PATH` 尚未包含 `~/.local/bin`，在 `~/.zprofile` 中加入 `export PATH="$HOME/.local/bin:$PATH"`，再打开新终端。此前也可用 `~/.local/bin/kev-ane` 执行下列命令。

模型文件默认放在**当前克隆仓库的 `hf-model/` 目录**。安装脚本先检查本地模型，文件齐全就跳过下载；否则先核对[模型仓库](https://huggingface.co/flylcw/KevANE-0.6B)的文件清单，再将所需文件下载到 `hf-model/`。若模型已放在其他目录，可运行 `bash scripts/install.sh --model-dir /模型目录的绝对路径`，无需复制或下载。无法直连 huggingface.co 的网络可先设置镜像，例如 `export HF_ENDPOINT=https://hf-mirror.com`；安装脚本的仓库检查和 `hf download` 都会遵循该变量。

运行环境由**本仓库**定义；所需的 Kev 兼容模块已按固定版本放在 `src/kev/`，普通用户无需另行 clone 或安装 Kev。Core ML 模型、分词器和 Pointer Head 从[独立的模型仓库](https://huggingface.co/flylcw/KevANE-0.6B)下载。安装后请保留克隆的源码目录，服务会从中读取代码和模型。

## 启动、停止与卸载

以下命令可在任意目录执行，无须先激活 Conda：

```bash
kev-ane start
kev-ane status
kev-ane logs
kev-ane stop
kev-ane restart
kev-ane cache status
```

KevANE 使用 macOS 当前用户的 LaunchAgent；执行 `start` 时才按需加载并立即返回，**不会随登录自动启动**。需要在继续操作前等接口就绪时，使用 `kev-ane start --wait`；`status` 会区分“加载中”和“已就绪”。服务监听 `127.0.0.1:8008`；`stop` 会等待 HTTP 和 Core ML 子进程退出。用户无须执行转换或编译命令：下载的模型直接交给 Core ML 运行。首次在一台 Mac 上加载时，Core ML 会自动完成一次设备准备，可能短时占满一个 CPU 核心；KevANE 把结果保存在 Git 忽略的 `build/compiled/`，后续启动不再重复。服务停止后可用 `kev-ane cache clear` 删除额外缓存，但下次启动需要重新准备。服务采用较低的 CPU 优先级并限制 CPU 库线程，以减少对前台应用的干扰。

检查服务：

```bash
curl --noproxy '*' http://127.0.0.1:8008/healthz
```

卸载已安装的命令和服务：

```bash
kev-ane uninstall
```

卸载会停止服务，移除命令、LaunchAgent 状态及编译缓存；只有安装脚本新建的 Conda 环境才会被删除。源码和模型默认保留。`kev-ane uninstall --remove-model` 只会删除**由安装脚本下载**的模型，不会删除预先放入 `hf-model/` 或通过 `--model-dir` 指定的模型。更新已克隆的仓库时，拉取代码、重新运行 `bash scripts/install.sh`，再执行 `kev-ane restart`。不再需要源码时可自行删除仓库目录。如需以前台方式运行，在仓库目录激活 `kevane-runtime` 后执行 `python scripts/30_systemone_server.py --model-dir hf-model --port 8008`。

## 连接 Jarvis

在 Jarvis 0.6.0 中，将 TypeSafe/System One 地址设为 `http://127.0.0.1:8008`、模型设为 `kevane-0.6b`；配置值见 [`integrations/jarvis/env.example`](integrations/jarvis/env.example)。先运行 `kev-ane start --wait`，再在 Jarvis 中测试连接。无需修改 Jarvis 安装包。

KevANE 只负责**判断和候选排序**。回复文字由 Jarvis 单独配置的生成服务提供；窗口、OCR、文字填入也由 Jarvis 自己管理。Jarvis 0.6.0 的悬浮窗只监测微信窗口；切换到 Codex 时会按设计隐藏，判断接口正常也不会让 Jarvis 识别 Codex 窗口或在本机生成回复。即使 KevANE 没有开启鉴权，Jarvis 的 TypeSafe 密钥也要填 `local`：Jarvis 用非空密钥选择这条判断链路。如设置 `KEVANE_API_KEY`，则改填相同值。

## 工作方式

```text
System One 请求 → Kev 编码 → Core ML 主干 → CPU Pointer Head → 判断结果
                          （FP16，每次最多 512 token）    （选择/评分/是非）
```

Core ML 模型在独立子进程中运行，以便原生 Core ML 意外退出时保住 HTTP 服务。合并后的 PyTorch 主干仅供转换，日常推理不加载。模型文件清单见 [Hugging Face 模型卡](https://huggingface.co/flylcw/KevANE-0.6B)。

## 验证结果与限制

- 当前 Core ML 图的输入固定为 512 token，是首版针对短判断和较低延迟选定的转换参数，不是 Kev 模型的完整上下文上限。打包请求超过 512 token 时，服务会把最多 16 个相互独立的问题拆成多次推理，每次都带上共同的上下文。如果“上下文 + 任意单个问题”仍超过 512 token，就返回 HTTP 422；不会悄悄截断。聊天历史过长时可能损失有用信息，要彻底扩大单次上下文需重新转换并验证更长的模型，多次推理也会增加延迟。
- 与原始 FP32 路径的记录对比中，七个样本的 hidden-state 余弦相似度最低为 **0.999058**；**35 个判断中的 34 个**保持相同首选答案，另一个是概率接近的边界样本。该对比已在发布的 `kevane-runtime` 环境中复测确认。详见[对照记录](benchmarks/results/coreml_fp16_parity.json)。
- 模型就位后，可运行 `conda activate kevane-runtime && python -m pytest tests/` 执行同一套对照、契约和编码检查。新克隆的仓库在 `hf-model/` 尚未就绪时会自动跳过依赖模型的测试。
- `CPU_AND_NE` 排除 Core ML 主干使用 GPU，但不能证明每个操作都在 ANE 上执行；Pointer Head 和输入准备仍用 CPU。如需硬件归因，应使用 Instruments。
- 服务默认无鉴权且只适用于本机，不应直接暴露到网络。

## 从原始检查点重新转换

重建为可选步骤，比推理需要更多内存和空间。[`environment/build.yml`](environment/build.yml) 固定 PyTorch 合并环境；[`environment/conversion.yml`](environment/conversion.yml) 是单独的 Core ML 转换环境。两者均使用本仓库包含的 Kev 代码。构建时会下载上游 [Kev 检查点](https://huggingface.co/jaredpalmer/kev-0.6b)与 [Qwen 基座](https://huggingface.co/Qwen/Qwen3-0.6B-Base)权重。

```bash
conda env create -f environment/build.yml
conda activate kevane-build
python -m pip install -e .
python scripts/10_merge_lora.py
conda deactivate

conda env create -f environment/conversion.yml
conda activate kevane-conversion
python -m pip install -e .
git clone https://github.com/anemll/anemll.git third_party/anemll
git -C third_party/anemll checkout f4ad26d061dc2e426faa19f81c367343bdeb9f0d
python -m pip install --no-deps -e third_party/anemll
python scripts/25_prepare_model_repo.py
python scripts/21_convert_anemll_backbone.py --seq-len 512
```

合并权重写入忽略发布的 `build/merged/`，转换产物写入 `hf-model/`。ANEMLL 仅用于转换，日常推理不依赖它。重新转换后，可在 `kevane-runtime` 环境执行 `python scripts/22_verify_coreml_hidden.py` 对照参考样本。

## 源码、模型与许可

源码仓库为 [LCW0NJUPT/KevANE](https://github.com/LCW0NJUPT/KevANE)，模型仓库为 [flylcw/KevANE-0.6B](https://huggingface.co/flylcw/KevANE-0.6B)。KevANE 与转换模型采用 Apache-2.0。内含的 Kev 模块版本见 [`src/kev/UPSTREAM.txt`](src/kev/UPSTREAM.txt)，来源说明见 [NOTICE](NOTICE)。KevANE 与 Kev、Qwen、ANEMLL 和 Jarvis 均为独立项目。

当前目录就是 GitHub 源码仓库，可直接从这里提交源码、脚本、Conda 环境和文档。较大的 Core ML 权重单独发布到 Hugging Face，并由 `.gitignore` 排除在 Git 提交之外。本机权重与模型卡一起放在 `hf-model/`，这也是安装脚本的默认模型目录。构建中间产物和其他本机文件同样由 `.gitignore` 排除。
