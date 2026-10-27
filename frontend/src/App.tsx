import React, { useState, useRef } from 'react';
import './styles/app.css';

const App: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [clinicalAnswer, setClinicalAnswer] = useState<string | null>(null);
  const [status, setStatus] = useState('Disconnected');
  const [error, setError] = useState<string | null>(null);
  const [textInput, setTextInput] = useState('');

  const socketRef = useRef<WebSocket | null>(null);
  const recordingCtxRef = useRef<AudioContext | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef<number>(0);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const getPlaybackContext = (): AudioContext => {
    if (!playbackCtxRef.current || playbackCtxRef.current.state === 'closed') {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      playbackCtxRef.current = new AudioCtx();
      nextPlayTimeRef.current = 0;
    }
    if (playbackCtxRef.current.state === 'suspended') {
      playbackCtxRef.current.resume();
    }
    return playbackCtxRef.current;
  };

  const connect = async () => {
    setError(null);
    setStatus('Connecting...');

    try {
      const socket = new WebSocket('ws://localhost:8000/ws/voice');
      socketRef.current = socket;

      socket.onopen = () => {
        setIsConnected(true);
        setStatus('Connected');
      };

      socket.onmessage = async (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'text') {
            setTranscript((prev) => [...prev, `AI: ${msg.text}`]);
          } else if (msg.type === 'audio') {
            playAudioChunk(msg.data);
          } else if (msg.type === 'clinical_answer') {
            setClinicalAnswer(msg.answer);
            setTranscript((prev) => [...prev, 'System: Clinical guidelines retrieved and verified.']);
          } else if (msg.type === 'error') {
            setError(msg.message);
          }
        } catch (e) {
          console.error('Failed to parse incoming WebSocket message', e);
        }
      };

      socket.onclose = () => {
        setIsConnected(false);
        setIsRecording(false);
        setStatus('Disconnected');
      };

      socket.onerror = (err) => {
        setError('WebSocket error occurred');
        console.error('WebSocket error:', err);
      };

    } catch (e) {
      setError('Failed to connect to server');
    }
  };

  const disconnect = () => {
    socketRef.current?.close();
    stopRecording();
    if (playbackCtxRef.current && playbackCtxRef.current.state !== 'closed') {
      playbackCtxRef.current.close();
      playbackCtxRef.current = null;
    }
  };

  const startRecording = async () => {
    try {
      // Warm up playback context on user gesture
      getPlaybackContext();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 16000 });
      recordingCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      // ScriptProcessor with bufferSize=4096, 1 input channel, 1 output channel
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        const inputData = e.inputBuffer.getChannelData(0);
        // Convert Float32 samples (-1.0 to 1.0) to signed 16-bit PCM little-endian
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Encode binary buffer to base64
        const uint8Bytes = new Uint8Array(pcmData.buffer);
        let binaryStr = '';
        for (let i = 0; i < uint8Bytes.length; i++) {
          binaryStr += String.fromCharCode(uint8Bytes[i]);
        }
        const base64Data = btoa(binaryStr);

        socketRef.current.send(JSON.stringify({
          type: 'audio',
          data: base64Data,
          sampleRate: 16000,
          format: 'pcm16',
        }));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
      setIsRecording(true);
      setStatus('Listening...');
    } catch (e) {
      setError('Microphone access denied or audio device unavailable');
      console.error(e);
    }
  };

  const stopRecording = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (recordingCtxRef.current && recordingCtxRef.current.state !== 'closed') {
      recordingCtxRef.current.close();
      recordingCtxRef.current = null;
    }
    setIsRecording(false);
    if (isConnected) {
      setStatus('Connected');
    }
  };

  const playAudioChunk = (base64Audio: string) => {
    try {
      const audioCtx = getPlaybackContext();
      const binaryString = atob(base64Audio);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Convert 16-bit PCM bytes to Float32
      const pcmData = new Int16Array(bytes.buffer);
      const floatData = new Float32Array(pcmData.length);
      for (let i = 0; i < pcmData.length; i++) {
        floatData[i] = pcmData[i] / 32768.0;
      }

      // Gemini Live output PCM is natively 24 kHz
      const buffer = audioCtx.createBuffer(1, floatData.length, 24000);
      buffer.copyToChannel(floatData, 0);

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);

      // Smooth chunk scheduling to prevent overlapping clicks/garble
      const now = audioCtx.currentTime;
      if (nextPlayTimeRef.current < now) {
        nextPlayTimeRef.current = now;
      }
      source.start(nextPlayTimeRef.current);
      nextPlayTimeRef.current += buffer.duration;
    } catch (err) {
      console.error('Error decoding/playing audio chunk:', err);
    }
  };

  const sendTextMessage = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!textInput.trim() || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    const message = textInput.trim();
    socketRef.current.send(JSON.stringify({
      type: 'text',
      text: message,
    }));
    setTranscript((prev) => [...prev, `User: ${message}`]);
    setTextInput('');
  };

  const normalizedStatusClass = status.toLowerCase().replace(/[^a-z]/g, '');

  return (
    <div className="app-container">
      <header>
        <h1>Clinical Knowledge Assistant</h1>
        <div className={`status-badge ${normalizedStatusClass}`}>
          {status}
        </div>
      </header>

      <main>
        <div className="control-panel">
          {!isConnected ? (
            <button onClick={connect} className="btn-primary">Connect to Assistant</button>
          ) : (
            <>
              <div className="voice-controls">
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`btn-voice ${isRecording ? 'recording' : ''}`}
                >
                  {isRecording ? 'Stop Listening' : 'Start Speaking'}
                </button>
                <button onClick={disconnect} className="btn-secondary">Disconnect</button>
              </div>

              <form onSubmit={sendTextMessage} className="text-input-panel">
                <input
                  type="text"
                  placeholder="Or type a greeting or clinical question..."
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                />
                <button type="submit">Send</button>
              </form>
            </>
          )}
        </div>

        {error && <div className="error-banner">{error}</div>}

        <div className="chat-layout">
          <div className="transcript-area">
            <h3>Conversation</h3>
            <div className="transcript-list">
              {transcript.length === 0 ? (
                <div className="placeholder">No messages yet. Connect and speak or type to start.</div>
              ) : (
                transcript.map((line, i) => (
                  <div key={i} className="transcript-line">{line}</div>
                ))
              )}
            </div>
          </div>

          <div className="clinical-answer-area">
            <h3>Clinical Guidance</h3>
            {clinicalAnswer ? (
              <div className="answer-card">
                <div className="answer-text">{clinicalAnswer}</div>
                <div className="review-flag">Clinician Review Required</div>
              </div>
            ) : (
              <div className="placeholder">No clinical guidelines retrieved yet.</div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;

