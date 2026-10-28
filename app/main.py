import asyncio
import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

from app.config import settings
from app.rag.pipeline import ClinicalAssistantOrchestrator
from app.voice.voice_bridge import VoiceBridge, CLINICAL_TOOL_DEFINITION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established with client")

    # Initialize Gemini Live client for this session
    client = genai.Client(api_key=settings.GEMINI_API_KEY, http_options={"api_version": "v1beta"})

    model_id = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")
    logger.info("Connecting to Gemini Live session with model: %s", model_id)

    try:
        connect_config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            tools=[CLINICAL_TOOL_DEFINITION],
            system_instruction=(
                "You are a Clinical Knowledge Assistant. You provide guidelines based on retrieved evidence. "
                "Always use the get_clinical_guidelines tool for medical and clinical queries. "
                "After receiving clinical guidelines from the tool, synthesize the key recommendations concisely and speak the answer aloud to the clinician. "
                "Be concise, clear, and professional."
            ),
        )

        async with client.aio.live.connect(model=model_id, config=connect_config) as session:
            logger.info("Gemini Live session successfully initialized and ready")

            async def receive_from_browser():
                """Reads audio/text from browser and sends to Gemini via real-time inputs."""
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

                                # Send raw 16kHz mono PCM bytes via documented Live API streaming method
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=pcm_bytes,
                                        mime_type=f"audio/pcm;rate={sample_rate}",
                                    )
                                )
                            except Exception:
                                logger.exception("Failed to process or forward incoming audio chunk to Gemini")

                        elif msg_type == "text":
                            query_text = message.get("text", "")
                            logger.info("Received text message from browser (length: %d chars)", len(query_text))
                            await session.send_realtime_input(text=query_text)

                        else:
                            logger.warning("Unknown message type received from browser: %s", msg_type)

                except WebSocketDisconnect:
                    logger.info("Browser disconnected from receive loop")
                except Exception:
                    logger.exception("Error in receive_from_browser loop")

            async def send_to_browser():
                """Reads responses from Gemini and sends to browser."""
                try:
                    async for response in session.receive():
                        # Handle user speech transcription and assistant outputs
                        if response.server_content:
                            # Forward user speech transcription if available
                            if response.server_content.interim_input_transcription and response.server_content.interim_input_transcription.text:
                                await websocket.send_json({
                                    "type": "user_transcription",
                                    "text": response.server_content.interim_input_transcription.text,
                                    "is_final": False,
                                })

                            if response.server_content.input_transcription and response.server_content.input_transcription.text:
                                await websocket.send_json({
                                    "type": "user_transcription",
                                    "text": response.server_content.input_transcription.text,
                                    "is_final": True,
                                })

                            # Forward output transcript if available
                            if response.server_content.output_transcription and response.server_content.output_transcription.text:
                                await websocket.send_json({
                                    "type": "text",
                                    "text": response.server_content.output_transcription.text,
                                })

                            # Handle audio and text parts from model turn
                            if response.server_content.model_turn:
                                parts = response.server_content.model_turn.parts
                                for part in parts:
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

                        # Handle Tool Calls
                        if response.tool_call and response.tool_call.function_calls:
                            for call in response.tool_call.function_calls:
                                call_name = call.name
                                call_id = call.id
                                args = call.args or {}
                                logger.info("Executing function call: %s (id=%s)", call_name, call_id)

                                if call_name == "get_clinical_guidelines":
                                    clinical_question = args.get("question", "")

                                    # Confirm user transcription from the tool call question if voice mode is active
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

                                    # Send tool response back to Gemini using documented function_responses API
                                    fn_response = types.FunctionResponse(
                                        id=call_id,
                                        name=call_name,
                                        response={"result": result},
                                    )
                                    await session.send_tool_response(function_responses=[fn_response])
                                    logger.info("Tool response sent back to Gemini Live for id=%s", call_id)

                                    # Push clinical answer to UI for rich card rendering
                                    await websocket.send_json({
                                        "type": "clinical_answer",
                                        "answer": result,
                                        "metadata": {
                                            "tool_used": call_name,
                                            "status": "completed",
                                        },
                                    })

                except WebSocketDisconnect:
                    logger.info("Browser disconnected from send loop")
                except Exception:
                    logger.exception("Error in send_to_browser loop")

            # Run both loops concurrently
            await asyncio.gather(receive_from_browser(), send_to_browser())

    except Exception:
        logger.exception("Gemini Live session failed during connection or execution")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Gemini Live session encountered an error and closed.",
            })
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

