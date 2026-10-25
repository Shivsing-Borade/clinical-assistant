import logging
import asyncio
import websockets
import json
from typing import Optional, Callable, AsyncGenerator
from app.config import settings

logger = logging.getLogger(__name__)

class GeminiLiveTranscriber:
    """
    Handles the WebSocket connection to Gemini Live for real-time transcription.
    Acts as a bridge between the client's audio stream and the Gemini API.
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.TRANSCRIPTION_MODEL
        self.uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={self.api_key}"
        self.websocket = None
        self._stop_event = asyncio.Event()

    async def connect(self):
        """Establishes the WebSocket connection to Gemini Live."""
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info(f"Connected to Gemini Live transcription service: {self.model}")

            # Initial setup message to configure the session
            setup_msg = {
                "setup": {
                    "model": f"models/{self.model}",
                }
            }
            await self.websocket.send(json.dumps(setup_msg))

            # Wait for setup confirmation
            response = await self.websocket.recv()
            logger.debug(f"Gemini setup response: {response}")

        except Exception as e:
            logger.error(f"Failed to connect to Gemini Live: {e}")
            raise

    async def stream_audio(self, audio_generator: AsyncGenerator[bytes, None],
                         transcription_callback: Callable[[str, bool], None]):
        """
        Streams audio chunks to Gemini and pushes transcripts back via callback.

        Args:
            audio_generator: Async generator yielding raw 16-bit PCM, mono, 16kHz audio chunks.
            transcription_callback: Callback function (text, is_final) to update the UI.
        """
        if not self.websocket:
            await self.connect()

        async def receive_transcripts():
            """Background task to process incoming transcripts from Gemini."""
            try:
                while not self._stop_event.is_set():
                    message = await self.websocket.recv()
                    data = json.loads(message)

                    # Gemini Live sends content in 'serverContent' blocks
                    if "serverContent" in data:
                        content = data["serverContent"]
                        if "modelTurn" in content:
                            turn = content["modelTurn"]
                            if "parts" in turn:
                                for part in turn["parts"]:
                                    if "text" in part:
                                        text = part["text"]
                                        # is_final can be inferred from the turn completion or specific flags
                                        # For transcription, we treat most as interim until the turn ends.
                                        transcription_callback(text, False)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Gemini Live connection closed while receiving transcripts.")
            except Exception as e:
                logger.error(f"Error receiving transcripts: {e}")

        # Start receiving transcripts in the background
        receiver_task = asyncio.create_task(receive_transcripts())

        try:
            async for chunk in audio_generator:
                if self._stop_event.is_set():
                    break

                # Gemini expects audio in base64 encoded 'realtimeInput' blocks
                import base64
                audio_payload = {
                    "realtimeInput": {
                        "mediaChunks": [
                            {
                                "mimeType": "audio/pcm",
                                "data": base64.b64encode(chunk).decode('utf-8')
                            }
                        ]
                    }
                }
                await self.websocket.send(json.dumps(audio_payload))

        except Exception as e:
            logger.error(f"Error streaming audio to Gemini: {e}")
        finally:
            self._stop_event.set()
            await receiver_task
            await self.disconnect()

    async def disconnect(self):
        """Closes the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from Gemini Live transcription service.")

    def stop(self):
        """Signals the streaming process to stop."""
        self._stop_event.set()
