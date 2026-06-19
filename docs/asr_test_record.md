whisper系列测试
python auralis_lab/asr.py --engine faster_whisper --model models/asr/faster-whisper-small --audio samples/asr/Tencent_test.wav
[2026-06-15 11:11:43.975] [ctranslate2] [thread 830461] [warning] The compute type inferred from the saved model is float16, but the target device or backend do not support efficient float16 computation. The model weights have been automatically converted to use the float32 compute type instead.
language=zh probability=1.000
ASR_TEXT:
你就是那个爱打篮球的人总理对任何事情都要抛根问题渐渐的他还真就睡著了这身衣服就像被淡雨淋过似的他为儿子买了一整根干著市区的停车收费将大幅提高他醒来后发现自己脸上有黑眼圈大风刮倒了一处再建厂房表大也觉得车夫的想法满有道理汹涌的河水顺流而下流得很快坚持终于让他有所收获据说这是当地最古老的小区


python auralis_lab/asr.py --engine faster_whisper --model models/asr/faster-whisper-medium --audio samples/asr/Tencent_test.wav
[2026-06-15 11:12:32.320] [ctranslate2] [thread 830983] [warning] The compute type inferred from the saved model is float16, but the target device or backend do not support efficient float16 computation. The model weights have been automatically converted to use the float32 compute type instead.
language=zh probability=1.000
ASR_TEXT:
你就是那个爱打篮球的人总理对任何事情都要刨根问底渐渐地他还真就睡著了这身衣服就像被淡雨淋过似的他为儿子买了一整根甘蔗市区的停车收费将大幅提高他醒来后发现自己脸上有黑眼圈大风刮倒了一处再建厂房表大爷觉得车夫的想法蛮有道理汹涌的河水顺流而下流得很快坚持终于让他有所收获据说这是当地最古老的小区


python auralis_lab/asr.py --engine faster_whisper --model models/asr/faster-whisper-large-v3 --audio samples/asr/Tencent_test.wav
[2026-06-15 11:14:05.564] [ctranslate2] [thread 831817] [warning] The compute type inferred from the saved model is float16, but the target device or backend do not support efficient float16 computation. The model weights have been automatically converted to use the float32 compute type instead.
language=zh probability=1.000
ASR_TEXT:
你就是那个爱打篮球的人总理对任何事情都要刨根问底渐渐的他还真就睡着了这身衣服就像被大雨淋过似的他为儿子买了一整根甘蔗市区的停车收费将大幅提高他醒来后发现自己脸上有黑眼圈大风刮倒了一处再见厂房表大爷觉得车夫的想法蛮有道理汹涌的河水顺流而下流得很快坚持终于让他有所收获据说这是当地最古老的小区

funASR系列测试
python auralis_lab/asr.py --engine funasr --model models/asr/funasr-paraformer-zh --punc-model models/asr/funasr-ct-punc --audio samples/asr/Tencent_test.wav
funasr version: 1.3.9.
WARNING:root:trust_remote_code: False
WARNING:root:trust_remote_code: False
rtf_avg: 0.037: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.17s/it]
rtf_avg: -0.146: 100%|█████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.73it/s]
ASR_TEXT:
你就是那个爱打篮球的人，总理对任何事情都要刨根问底。渐渐的，他还真就睡着了，这身衣服就像被大雨淋过似的，他为儿子买了一整根甘蔗市区的停车收费将大幅提高。他醒来后发现自己脸上有黑眼圈，大风刮倒了一处再建厂房表大爷觉得车夫的想法蛮有道理，汹涌的河水顺流而下，流的很快坚持，终于让他有所收获。据说这是当地最古老的小区。


SenseVoiceSmall测试
python auralis_lab/asr.py --engine sensevoice --model models/asr/sensevoice-small --audio samples/asr/Tencent_test.wav
funasr version: 1.3.9.
WARNING:root:trust_remote_code: False
rtf_avg: 0.010: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.03it/s]
ASR_TEXT:
你就是那个爱打篮球的人，总理对任何事情都要刨根问底，渐渐的他还真就睡着了，这身衣服就像被大雨淋过似的。他为儿子买了一整根甘蔗，市区的停车收费将大幅提高。他醒来后发现自己脸上有黑眼圈。大风刮倒了一处在建厂房。表大爷觉得车夫的想法蛮有道理，汹涌的河水顺流而下，流得很快，坚持终于让他有所收获，据说这是当地最古老的小区。。


sherpa-onnx + sensevoice测试
python auralis_lab/asr.py --engine sherpa_onnx --sherpa-model-type sensevoice --model models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 --audio samples/asr/Tencent_test.wav
ASR_TEXT:
你就是那个爱打篮球的人，总理对任何事情都要刨根问底，渐渐的他还真就睡着了，这身衣服就像被大雨淋过似的。他为儿子买了一整根甘蔗，市区的停车收费将大幅提高。他醒来后发现自己脸上有黑眼圈，大风刮倒了一处在建厂房。表大爷觉得车夫的想法蛮有道理，汹涌的河水顺流而下，流得很快，坚持终于让他有所收获。据说这是当地最古老的小区。。




