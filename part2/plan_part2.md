### Pipeline A：前沿特征解析路线（强烈推荐）
**架构：VGGT4D + ProPainter**
如果你对 Vision Transformer 的注意力机制以及深层特征的挖掘感兴趣，这条路线非常适合。
* [cite_start]**掩码提取 (VGGT4D)**：这是一个免训练（training-free）的方案，它通过计算 Vision Transformer 注意力图中的 Gram 相似度（Gram Similarity）来挖掘运动线索，从而实现零样本（zero-shot）分割 [cite: 82, 83]。
* [cite_start]**图像修复 (ProPainter)**：结合双域传播（光流 + 特征）和稀疏 Transformer，能非常有效地利用视频的时间冗余来填补大面积的遮挡 [cite: 87, 88]。
* [cite_start]**优势**：这种方法不依赖于传统的光流和单纯的边界框，而是直接从高维特征空间中提取动态信息，技术上非常优雅，也能为你后续（比如 Part 3）移植到其他 3D 基础模型打下很好的理论基础 [cite: 96, 97]。

### Pipeline B：工业级全能路线
**架构：SAM 2 + ProPainter**
如果你追求极致的掩码生成质量和运行效率，这条路线最为稳妥。
* [cite_start]**掩码提取 (SAM 2)**：Meta 的统一基础模型，它将 SAM 的提示分割能力扩展到了视频领域，并且具备实时运行的性能 [cite: 80, 81]。
* [cite_start]**图像修复 (ProPainter)**：同样负责处理大面积遮挡并重建清晰的背景纹理 [cite: 86, 88, 90]。
* [cite_start]**优势**：SAM 2 本身就代表了当前分割领域的顶级水平。它的介入能让你在后续计算 IoU (JM) 和 IoU (JR) 等指标时占据极大优势 [cite: 111, 112]。
