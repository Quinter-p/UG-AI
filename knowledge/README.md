# 凡人流修仙 AI 知识库 v1

用途：给本地 AI / RAG 系统提供一个“凡人修仙风格”的世界观、术语、人设和回答规则知识库。

重要说明：
- 本知识库是二次整理，不包含小说正文，不用于替代原著阅读。
- 适合做“修仙界助手 / 器灵 / 宗门顾问 / 苟道规划器”，不建议做逐章剧情复述机器人。
- 如果继续扩展，请优先加入你自己写的总结、表格、术语解释、人物关系，不要复制原文段落。

建议导入顺序：
1. `00_system_prompt.md`
2. `01_worldview.md`
3. `02_cultivation_levels.md`
4. `03_ai_behavior_rules.md`
5. `04_characters_core.md`
6. `05_resources_items.md`
7. `06_factions_places.md`
8. `07_timeline_summary.md`
9. `09_terms_glossary.md`
10. `08_qa_templates.md`

RAG 切块建议：
- 每块 300～800 中文字。
- 标题、标签、来源说明保留。
- 不要把所有内容塞进一个超长文件。
- 查询时优先召回 `00_system_prompt.md` 和 `03_ai_behavior_rules.md`，这样 AI 的说话方式会更稳定。
