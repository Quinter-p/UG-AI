import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

# ====== 配置 ======
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_FILE = "record.wav"

# Whisper 模型大小：tiny / base / small / medium
# 你显卡 12G，可以先用 small
WHISPER_MODEL_SIZE = "small"


def record_audio():
    print(f"开始录音，请说话，录音时长 {RECORD_SECONDS} 秒...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()
    
    write(AUDIO_FILE, SAMPLE_RATE, audio)
    print(f"录音完成，已保存为 {AUDIO_FILE}")


def transcribe_audio():
    print("正在加载 Whisper 模型...")

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cuda",
        compute_type="float16"
    )

    print("正在识别语音...")

    segments, info = model.transcribe(
        AUDIO_FILE,
        language="zh",
        beam_size=5
    )

    text = ""

    for segment in segments:
        text += segment.text

    text = text.strip()

    print("\n识别结果：")
    print(text)

    print("\n检测语言：", info.language)
    print("语言概率：", info.language_probability)


if __name__ == "__main__":
    record_audio()
    transcribe_audio()