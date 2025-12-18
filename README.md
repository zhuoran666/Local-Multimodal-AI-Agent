这是一份为您定制的 `README.md` 文档。它是根据您提供的完整 Python 代码逻辑、所使用的技术栈以及命令行接口编写的。您可以直接将以下内容复制并保存为 `README.md` 文件。

---

# 🧠 Local AI Multimodal Manager (本地 AI 智能文献与图像管理助手)

一个功能强大的本地化多模态资源管理工具。它利用最前沿的 AI 模型（CLIP, Transformer, BART），实现了对**图片**和**学术论文（PDF）**的语义检索、深度内容分析以及自动化分类整理。所有数据和模型均存储在本地，保护隐私且无需联网即可使用（首次加载除外）。

## ✨ 核心功能 (Core Features)

1.  **🖼️ 语义化以文搜图 (Image Semantic Search)**
    *   支持使用自然语言（中文或英文）搜索本地图片库。
    *   无需标签，AI 自动理解图片内容（例如搜索 "一只在睡觉的猫" 即可找到对应图片）。
2.  **📄 深度文档检索 (Deep Document Retrieval - RAG)**
    *   **片段级定位**：不仅仅是找到文档，还能精准定位到 PDF 的具体**页码**和**文本片段**。
    *   **跨语言检索**：支持用中文搜索英文论文（反之亦然）。
    *   **抗干扰读取**：基于 `pdfplumber` 的智能文本提取，有效处理单词粘连和排版问题。
3.  **📂 智能自动化分类 (Zero-Shot Organization)**
    *   **零样本分类**：无需预先训练，只需提供类别名称（如 "CV, NLP, 金融"），AI 即可根据论文的**标题**和**摘要**自动将文件移动到对应文件夹。
    *   支持单个文件或整个文件夹的批量整理。
4.  **🔒 完全本地化**
    *   向量数据库 (`ChromaDB`) 和所有 AI 模型均存储在本地 (`./db` 和 `./model_cache`)，无隐私泄露风险。

---

## 🛠️ 技术选型 (Tech Stack)

本项目集成了多个 SOTA (State-of-the-Art) 模型与库，以实现最佳性能：

| 模块 | 技术/模型 | 说明 |
| :--- | :--- | :--- |
| **图像编码** | **OpenAI CLIP (ViT-B/32)** | 将图像转换为向量，实现图文对齐。 |
| **图像检索** | **Multilingual-CLIP** | `clip-ViT-B-32-multilingual-v1`，支持多语言文本搜图。 |
| **文档嵌入** | **Sentence-Transformer** | `paraphrase-multilingual-MiniLM-L12-v2`，强大的多语言文本向量化模型。 |
| **智能分类** | **BART-Large-MNLI** | `facebook/bart-large-mnli`，用于 Zero-Shot（零样本）文本分类。 |
| **数据库** | **ChromaDB** | 高性能本地向量数据库，支持余弦相似度检索。 |
| **PDF 处理** | **pdfplumber** | 专业的 PDF 文本提取库，解决复杂排版解析问题。 |

---

## ⚙️ 环境配置与安装 (Installation)

建议使用 Python 3.8+ 环境。

### 1. 克隆或下载代码
确保 `main.py` 在你的工作目录中。

### 2. 安装依赖库
请在终端运行以下命令安装所需依赖：

```bash
# 基础依赖
pip install torch torchvision
pip install chromadb transformers sentence-transformers pdfplumber pillow

# 安装 OpenAI CLIP (需从 GitHub 直接安装)
pip install git+https://github.com/openai/CLIP.git
```

### 3. 目录结构说明
运行程序后，系统会自动创建以下文件夹：
*   `./model_cache`: 存放下载的 AI 模型文件（避免重复下载）。
*   `./db`: 存放 ChromaDB 向量数据库文件。

---

## 🚀 使用说明 (Usage)

所有操作均通过命令行 (`main.py`) 完成。

### 1. 图片管理

#### 添加图片到数据库
支持添加单张图片或整个文件夹（自动递归扫描）。
```bash
# 添加单张图片
python main.py add_image ./photos/cat.jpg

# 批量添加文件夹中的所有图片
python main.py add_image E:/Datasets/Images
```

#### 以文搜图
使用自然语言描述你想找的图片。
```bash
# 搜索示例
python main.py search_image "雪山下的湖泊"
# 或者
python main.py search_image "a dog playing with a ball"
```

---

### 2. 文档（PDF）管理

#### 添加文档到数据库
系统会对 PDF 进行全文切片（Chunking），并记录页码。支持单文件或文件夹。
```bash
# 索引单个 PDF
python main.py add_doc ./papers/Attention_is_all_you_need.pdf

# 索引整个论文文件夹
python main.py add_doc E:/Research/Papers
```

#### 语义搜索文档
搜索文档的具体内容，返回相关片段、文件名及页码。
```bash
# 提问式搜索
python main.py search_doc "Transformer 的核心架构是什么？"

# 关键词搜索
python main.py search_doc "residual learning vanishing gradient"
```
> **输出示例：**
> ```text
> 来源: Attention is All you Need.pdf (第 3 页)
> 置信度: 0.5939
> 片段内容: "...Figure 1: The Transformer - model architecture..."
> ```

---

### 3. 智能文件整理 (Sort)

无需预先定义规则，只需告诉 AI 你想要的类别，它会自动分析文件内容并归类整理。

#### 核心逻辑
AI 会读取文件的**文件名**和**摘要**，结合你输入的类别进行推理。

#### 命令格式
```bash
python main.py sort <目标路径> --topics "<类别1>,<类别2>,<类别3>"
```

#### 示例
假设你有一堆乱七八糟的论文在 `./downloads` 文件夹下：

```bash
# 自动将论文分类到 CV, NLP, RL 三个文件夹中
python main.py sort ./downloads --topics "Computer Vision, Natural Language Processing, Reinforcement Learning"
```

> **💡 提示：** 为了提高分类准确率，建议使用**全称**（如 "Computer Vision" 而非 "CV"），或者在类别中加入相关关键词。

---

## 📝 注意事项 (Notes)

1.  **首次运行时间**：第一次运行时，程序会自动从 HuggingFace 和 OpenAI 下载所需的模型权重（约 2GB+）。请保持网络通畅，下载完成后模型会保存在 `./model_cache`，后续运行速度会非常快。
2.  **显存占用**：程序会自动检测 GPU。如果有 NVIDIA 显卡 (CUDA)，速度会显著提升；如果是 CPU 运行，处理大量 PDF 时可能会稍慢。
3.  **重置数据库**：如果需要清空数据重新索引，只需直接删除目录下的 `./db` 文件夹即可。

---

## 📧 联系与反馈
如有问题，请提交 Issue 或检查代码中的报错日志。Enjoy your AI Assistant! 🚀
