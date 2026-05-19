def build_reflection_memory_fragment(reflection_memory_text):
    text=str(reflection_memory_text or '').strip() or '暂无反思记忆。'
    return text + '\n\n以上是你从过往互动中形成的反思记忆。它用于改进长期行为风格，只在相关时自然参考，不要直接背诵。'
