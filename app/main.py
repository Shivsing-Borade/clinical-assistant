import asyncio
import base64
import logging
import os
import socket as _socket
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

from app.config import settings
from app.rag.pipeline import ClinicalAssistantOrchestrator
from app.voice.voice_bridge import VoiceBridge, CLINICAL_TOOL_DEFINITION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bounded retry config for Gemini Live WebSocket handshake timeouts.
# DNS failures (gaierror) are NOT retried — they are environment-level problems.
_MAX_HANDSHAKE_RETRIES = 2       # max extra attempts after first failure
_HANDSHAKE_RETRY_DELAY = 1.0    # base delay in seconds (multiplied by attempt number)

app = FastAPI(title="Clinical Assistant Voice API")

# CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator for tool use
orchestrator = ClinicalAssistantOrchestrator()
bridge = VoiceBridge(orchestrator)


def decode_audio_message(message: Dict[str, Any]) -> bytes:
    """Decodes incoming browser audio payload (base64 string or raw bytes) to PCM bytes."""
    raw_data = message.get("data")
    if raw_data is None:
        raise ValueError("Audio message missing 'data' field")
    if isinstance(raw_data, bytes):
        return raw_data
    if isinstance(raw_data, str):
        return base64.b64decode(raw_data)
    raise TypeError(f"Unsupported audio data type: {type(raw_data)}")


def serialize_audio_data(audio_data: Any) -> str:
    """Serializes binary audio data (e.g. from Gemini Live inline_data.data) to base64 string."""
    if isinstance(audio_data, bytes):
        return base64.b64encode(audio_data).decode("utf-8")
    if isinstance(audio_data, str):
        return audio_data
    raise TypeError(f"Unsupported audio data type for serialization: {type(audio_data)}")


async def _send_error_to_browser(websocket: WebSocket, message: str) -> None:
    """Sends an error JSON message to the browser and closes the WebSocket gracefully."""
    try:
        await websocket.send_json({"type": "error", "message": message})
        await websocket.close()
    except Exception:
        pass  # Browser may already be disconnected


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established with client")

    client = genai.Client(api_key=settings.GEMINI_API_KEY, http_options={"api_version": "v1beta"})
    model_id = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
    logger.info("Connecting to Gemini Live session with model: %s", model_id)

    connect_config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=[CLINICAL_TOOL_DEFINITION],
        system_instruction=(
            "You are a Clinical Knowledge Assistant. You provide guidelines based on retrieved evidence. "
            "Always use the get_clinical_guidelines tool for medical and clinical queries. "
            "After receiving clinical guidelines from the tool, synthesize the key recommendations concisely "
            "and speak the answer aloud to the clinician. "
            "Be concise, clear, and professional."
        ),
    )

    # --- Bounded retry loop for connection handshake timeouts ---
    # DNS failures (gaierror / OSError) abort immediately — they won't self-heal.
    # TimeoutError / asyncio.TimeoutError are retried with linear backoff.
    # All other errors propagate to the outer handler.
    for attempt in range(1, _MAX_HANDSHAKE_RETRIES + 2):  # attempts: 1, 2, 3
        try:
            async with client.aio.live.connect(model=model_id, config=connect_config) as session:
                logger.info("Gemini Live session ready (attempt %d)", attempt)
                await _run_voice_session(websocket, session)
            break  # Session completed normally — exit retry loop

        except (_socket.gaierror, OSError) as exc:
            # DNS resolution failed or network is unreachable — do not retry
            logger.error(
                "DNS/network error connecting to Gemini Live (attempt %d): %s",
                attempt, exc,
            )
            await _send_error_to_browser(
                websocket,
                "Cannot reach the Gemini Live API. Please check your network connection and try again.",
            )
            return

        except (TimeoutError, asyncio.TimeoutError) as exc:
            if attempt <= _MAX_HANDSHAKE_RETRIES:
                delay = _HANDSHAKE_RETRY_DELAY * attempt
                logger.warning(
                    "Gemini Live handshake timed out (attempt %d/%d) — retrying in %.1fs: %s",
                    attempt, _MAX_HANDSHAKE_RETRIES + 1, delay, exc,
                )
                await asyncio.sleep(delay)
                # Continue to next attempt
            else:
                logger.error(
                    "Gemini Live handshake timed out after %d attempts — giving up",
                    attempt,
                )
                await _send_error_to_browser(
                    websocket,
                    "Connection to the Gemini Live API timed out. Please try again in a moment.",
                )
                return

        except WebSocketDisconnect:
            logger.info("Browser disconnected before Gemini Live session could start")
            return

        except Exception:
            logger.exception("Gemini Live session failed (attempt %d)", attempt)
            await _send_error_to_browser(
                websocket,
                "Gemini Live session encountered an error and closed.",
            )
            return


async def _run_voice_session(websocket: WebSocket, session: Any) -> None:
    """
    Runs the bidirectional audio exchange between the browser and an active Gemini Live session.
    Terminates when either:
      - The browser disconnects.
      - The Gemini Live session ends (no more turns).
      - An unrecoverable error occurs in either direction.
    """

    async def receive_from_browser() -> None:
        """Reads audio/text/stop messages from the browser and forwards to Gemini Live."""
        try:
            while True:
                message = await websocket.receive_json()
                msg_type = message.get("type")

                if msg_type == "audio":
                    try:
                        pcm_bytes = decode_audio_message(message)
                        sample_rate = message.get("sampleRate", 16000)
                        logger.debug(
                            "Received audio chunk: %d bytes (rate=%s, format=%s)",
                            len(pcm_bytes),
                            sample_rate,
                            message.get("format", "pcm16"),
                        )
                        # Send raw 16kHz mono PCM via documented Live API streaming method
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=pcm_bytes,
                                mime_type=f"audio/pcm;rate={sample_rate}",
                            )
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to forward audio chunk to Gemini (session closed?): %s", exc
                        )
                        break

                elif msg_type == "text":
                    query_text = message.get("text", "")
                    logger.info(
                        "Received text message from browser (length: %d chars)", len(query_text)
                    )
                    await session.send_realtime_input(text=query_text)

                elif msg_type == "stop":
                    logger.info("Received stop signal from browser; turn cancelled by user")

                else:
                    logger.warning("Unknown message type received from browser: %s", msg_type)

        except WebSocketDisconnect:
            logger.info("Browser disconnected from receive loop")
        except asyncio.CancelledError:
            logger.info("receive_from_browser loop cancelled")
        except Exception:
            logger.exception("Error in receive_from_browser loop")
            raise

    async def send_to_browser() -> None:
        """
        Reads Gemini Live responses and forwards them to the browser.
        Supports multi-turn: the outer while-True loop restarts session.receive() for each turn.
        The loop exits when a turn completes with zero messages (session ended).
        """
        try:
            while True:
                turn_has_messages = False

                async for response in session.receive():
                    # Every response object is processed inside this loop — not outside it.
                    turn_has_messages = True

                    # --- Token usage metadata ---
                    if response.usage_metadata:
                        total_tokens = getattr(response.usage_metadata, "total_token_count", None)
                        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
                        response_tokens = getattr(response.usage_metadata, "response_token_count", None)
                        logger.info(
                            "Gemini Live usage: total=%s prompt=%s response=%s",
                            total_tokens, prompt_tokens, response_tokens,
                        )
                        if total_tokens is not None and total_tokens > 0:
                            await websocket.send_json({
                                "type": "usage_metadata",
                                "total_tokens": total_tokens,
                                "prompt_tokens": prompt_tokens,
                                "response_tokens": response_tokens,
                            })

                    # --- Server content (audio, text, transcriptions, turn signals) ---
                    if response.server_content:
                        sc = response.server_content

                        if sc.interrupted:
                            logger.info("Model turn interrupted by user speech")
                            await websocket.send_json({"type": "interrupted"})

                        if sc.turn_complete:
                            await websocket.send_json({"type": "turn_complete"})

                        # Interim user speech transcription (streaming, not final)
                        if (
                            sc.interim_input_transcription
                            and sc.interim_input_transcription.text
                        ):
                            await websocket.send_json({
                                "type": "user_transcription",
                                "text": sc.interim_input_transcription.text,
                                "is_final": False,
                            })

                        # Final user speech transcription
                        if sc.input_transcription and sc.input_transcription.text:
                            await websocket.send_json({
                                "type": "user_transcription",
                                "text": sc.input_transcription.text,
                                "is_final": True,
                            })

                        # Assistant output transcript (text form of spoken response)
                        if sc.output_transcription and sc.output_transcription.text:
                            await websocket.send_json({
                                "type": "text",
                                "text": sc.output_transcription.text,
                            })

                        # Audio and text parts from the model turn
                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    # Output audio is 24kHz PCM from Gemini Live
                                    b64_audio = serialize_audio_data(part.inline_data.data)
                                    await websocket.send_json({
                                        "type": "audio",
                                        "data": b64_audio,
                                        "sampleRate": 24000,
                                    })
                                if part.text:
                                    await websocket.send_json({
                                        "type": "text",
                                        "text": part.text,
                                    })

                    # --- Tool calls (function calling with BLOCKING behavior) ---
                    if response.tool_call and response.tool_call.function_calls:
                        for call in response.tool_call.function_calls:
                            call_name = call.name
                            call_id = call.id
                            args = call.args or {}
                            logger.info(
                                "Executing function call: %s (id=%s)", call_name, call_id
                            )

                            if call_name == "get_clinical_guidelines":
                                clinical_question = args.get("question", "")

                                # Echo the transcribed question back to the UI immediately
                                if clinical_question:
                                    await websocket.send_json({
                                        "type": "user_transcription",
                                        "text": clinical_question,
                                        "is_final": True,
                                    })

                                result = await bridge.get_clinical_guidelines(
                                    question=clinical_question,
                                    history=args.get("history"),
                                )

                                # Return result to Gemini Live via documented function_responses API
                                fn_response = types.FunctionResponse(
                                    id=call_id,
                                    name=call_name,
                                    response={"result": result},
                                )
                                await session.send_tool_response(
                                    function_responses=[fn_response]
                                )
                                logger.info(
                                    "Tool response sent back to Gemini Live for id=%s", call_id
                                )

                                # Push the structured clinical answer to the UI
                                await websocket.send_json({
                                    "type": "clinical_answer",
                                    "answer": result,
                                    "metadata": {
                                        "tool_used": call_name,
                                        "status": "completed",
                                    },
                                })

                # --- End of turn: check if the session delivered any messages ---
                if not turn_has_messages:
                    logger.info(
                        "Gemini Live session finished (receive() returned no messages)"
                    )
                    break
                # Otherwise loop back and wait for the next turn

        except WebSocketDisconnect:
            logger.info("Browser disconnected from send loop")
        except asyncio.CancelledError:
            logger.info("send_to_browser loop cancelled")
        except Exception:
            logger.exception("Error in send_to_browser loop")
            raise

    # Run both directions concurrently; cancel the other when one finishes
    receive_task = asyncio.create_task(receive_from_browser())
    send_task = asyncio.create_task(send_to_browser())

    done, pending = await asyncio.wait(
        [receive_task, send_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    for task in done:
        if not task.cancelled() and task.exception():
            exc = task.exception()
            if not isinstance(exc, WebSocketDisconnect):
                raise exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
