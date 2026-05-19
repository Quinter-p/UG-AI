def build_emotion_fragment(emotion):
    emotion = emotion or {}

    return (
        f"当前心情：{emotion.get('mood_text', '平静')} ({emotion.get('mood', 'calm')})\n"
        f"精神：{emotion.get('energy', 70)}/100\n"
        f"亲近度：{emotion.get('affection', 50)}/100\n"
        f"警觉度：{emotion.get('alertness', 30)}/100\n"
        "这些是内部状态，只能影响语气和反应倾向，不要把状态表直接念给用户。"
    )
