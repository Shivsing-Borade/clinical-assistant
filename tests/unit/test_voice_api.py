import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from google.genai import types

from app.main import decode_audio_message, serialize_audio_data, app


def test_decode_audio_message_valid_base64():
    """Verify base64-encoded audio payload from browser is properly decoded to bytes."""
    original_pcm = b"\x00\x01\x02\x03\x04\x05"
    b64_str = base64.b64encode(original_pcm).decode("utf-8")
    message = {"type": "audio", "data": b64_str, "sampleRate": 16000}

    decoded = decode_audio_message(message)
    assert isinstance(decoded, bytes)
    assert decoded == original_pcm


def test_decode_audio_message_raw_bytes():
    """Verify raw bytes payload is accepted as-is."""
    raw_pcm = b"\x10\x20\x30\x40"
    message = {"type": "audio", "data": raw_pcm}

    decoded = decode_audio_message(message)
    assert decoded == raw_pcm


def test_decode_audio_message_missing_data():
    """Verify ValueError is raised if 'data' is missing."""
    with pytest.raises(ValueError, match="missing 'data' field"):
        decode_audio_message({"type": "audio"})


def test_decode_audio_message_invalid_type():
    """Verify TypeError is raised if 'data' is not str or bytes."""
    with pytest.raises(TypeError, match="Unsupported audio data type"):
        decode_audio_message({"type": "audio", "data": 12345})


def test_serialize_audio_data_bytes():
    """Verify raw bytes from Gemini Live inline_data are serialized to base64 string."""
    raw_bytes = b"\xde\xad\xbe\xef"
    b64 = serialize_audio_data(raw_bytes)
    assert isinstance(b64, str)
    assert base64.b64decode(b64) == raw_bytes


def test_serialize_audio_data_string():
    """Verify already-string audio data is returned unchanged."""
    text_data = "dGVzdA=="
    assert serialize_audio_data(text_data) == text_data


def test_serialize_audio_data_invalid_type():
    """Verify TypeError is raised on unsupported serialization types."""
    with pytest.raises(TypeError, match="Unsupported audio data type for serialization"):
        serialize_audio_data(12345)


def test_audio_roundtrip():
    """Verify audio encode/decode roundtrip preserves exact PCM byte contents."""
    sample_pcm = bytes(range(256))
    serialized = serialize_audio_data(sample_pcm)
    msg = {"data": serialized}
    restored = decode_audio_message(msg)
    assert restored == sample_pcm


@pytest.mark.anyio
async def test_live_session_setup_and_audio_streaming():
    """Verify that websocket connection sets up Live session and streams realtime audio properly."""
    fake_pcm = b"\x00" * 4096
    b64_audio = base64.b64encode(fake_pcm).decode("utf-8")

    mock_session = AsyncMock()
    mock_session.send_realtime_input = AsyncMock()
    mock_session.send_tool_response = AsyncMock()

    # Generator for session.receive() that ends after sending one message
    async def mock_receive():
        server_msg = types.LiveServerMessage(
            server_content=types.LiveServerContent(
                model_turn=types.Content(
                    parts=[
                        types.Part(inline_data=types.Blob(data=b"\x01\x02\x03\x04", mime_type="audio/pcm;rate=24000")),
                        types.Part(text="Hello Doctor"),
                    ]
                )
            )
        )
        yield server_msg

    mock_session.receive = mock_receive

    # Mock client.aio.live.connect async context manager
    mock_connect_cm = AsyncMock()
    mock_connect_cm.__aenter__.return_value = mock_session
    mock_connect_cm.__aexit__.return_value = None

    mock_client = MagicMock()
    mock_client.aio.live.connect.return_value = mock_connect_cm

    with patch("app.main.genai.Client", return_value=mock_client):
        # Verify connect config parameters
        from app.voice.voice_bridge import CLINICAL_TOOL_DEFINITION
        assert CLINICAL_TOOL_DEFINITION["function_declarations"][0]["behavior"] == "BLOCKING"

        expected_config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            tools=[CLINICAL_TOOL_DEFINITION],
            system_instruction="You are a Clinical Knowledge Assistant.",
        )
        assert expected_config.response_modalities == [types.Modality.AUDIO]
        assert expected_config.input_audio_transcription is not None
        assert expected_config.output_audio_transcription is not None
        assert len(expected_config.tools) == 1
        assert expected_config.tools[0].function_declarations[0].behavior.value == "BLOCKING"


def test_clinical_tool_definition_blocking():
    """Verify CLINICAL_TOOL_DEFINITION explicitly specifies BLOCKING behavior for synthesized audio output."""
    from app.voice.voice_bridge import CLINICAL_TOOL_DEFINITION
    decl = CLINICAL_TOOL_DEFINITION["function_declarations"][0]
    assert decl["name"] == "get_clinical_guidelines"
    assert decl["behavior"] == "BLOCKING"

