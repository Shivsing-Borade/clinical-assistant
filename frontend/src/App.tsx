import React, { useState, useEffect, useRef } from 'react';
import './styles/app.css';

interface ClinicalData {
  answer: string;
  caution?: string;
  sources?: string[];
  clinicianReviewRequired?: boolean;
  rawText: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  text: string;
  timestamp: string;
  clinicalData?: ClinicalData;
  isTranscription?: boolean;
  isPartial?: boolean;
  isVoiceInteraction?: boolean;
  durationSec?: string;
  tokenCount?: number;
}

interface ChatThread {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}

const STORAGE_THREADS_KEY = 'clinical_assistant_threads_v1';
const STORAGE_ACTIVE_KEY = 'clinical_assistant_active_thread_id_v1';

function createNewThread(): ChatThread {
  return {
    id: 'thread_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
    title: 'New Consultation',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    messages: [],
  };
}

function loadInitialThreads(): { threads: ChatThread[]; activeId: string } {
  try {
    const raw = localStorage.getItem(STORAGE_THREADS_KEY);
    const storedActive = localStorage.getItem(STORAGE_ACTIVE_KEY);
    if (raw) {
      const parsed: ChatThread[] = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const activeExists = parsed.some((t) => t.id === storedActive);
        return {
          threads: parsed,
          activeId: activeExists && storedActive ? storedActive : parsed[0].id,
        };
      }
    }
  } catch (err) {
    console.error('Failed to load chat history from localStorage', err);
  }

  const initial = createNewThread();
  return {
    threads: [initial],
    activeId: initial.id,
  };
}

function parseClinicalAnswer(raw: string): ClinicalData {
  let answer = '';
  let caution: string | undefined;
  let sources: string[] | undefined;
  let clinicianReviewRequired = false;

  const lines = raw.split('\n');
  const answerLines: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('Answer:')) {
      answerLines.push(trimmed.replace(/^Answer:\s*/, ''));
    } else if (trimmed.startsWith('Caution:')) {
      caution = trimmed.replace(/^Caution:\s*/, '');
    } else if (trimmed.startsWith('Sources:')) {
      const srcText = trimmed.replace(/^Sources:\s*/, '');
      sources = srcText.split(',').map((s) => s.trim()).filter(Boolean);
    } else if (trimmed.toLowerCase().includes('clinician review')) {
      clinicianReviewRequired = true;
    } else if (trimmed) {
      answerLines.push(trimmed);
    }
  }

  answer = answerLines.join('\n').trim();
  if (!answer && raw) {
    answer = raw;
  }

  return {
    answer,
    caution,
    sources,
    clinicianReviewRequired,
    rawText: raw,
  };
}

function formatInline(str: string): React.ReactNode {
  const parts = str.split(/(\*\*[^*]+\*\*|__[^_]+__)/g);
  return parts.map((part, idx) => {
    if ((part.startsWith('**') && part.endsWith('**')) || (part.startsWith('__') && part.endsWith('__'))) {
      return <strong key={idx}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function renderFormattedText(text: string): React.ReactNode {
  if (!text) return null;
  const paragraphs = text.split(/\n\n+/);

  return (
    <>
      {paragraphs.map((para, pIdx) => {
        const lines = para.split('\n');
        const isBulletList = lines.every((l) => /^\s*([*•-]\s+|\d+\.\s+)/.test(l.trim()) || l.trim() === '');
        if (isBulletList && lines.some((l) => l.trim())) {
          return (
            <ul key={pIdx} className="message-list">
              {lines.filter((l) => l.trim()).map((l, lIdx) => {
                const cleanItem = l.trim().replace(/^([*•-]\s+|\d+\.\s+)/, '');
                return <li key={lIdx}>{formatInline(cleanItem)}</li>;
              })}
            </ul>
          );
        }

        return (
          <p key={pIdx} className="message-paragraph">
            {lines.map((l, lIdx) => (
              <React.Fragment key={lIdx}>
                {formatInline(l)}
                {lIdx < lines.length - 1 && <br />}
              </React.Fragment>
            ))}
          </p>
        );
      })}
    </>
  );
}

const STARTER_PROMPTS = [
  "What are the risk classes of IVD devices in India?",
  "What are the high-risk factors for severe dengue in children and elderly?",
  "When is platelet transfusion considered in dengue patients?",
  "How is the Minimum Inhibitory Concentration (MIC) defined in antimicrobial testing?",
];

const App: React.FC = () => {
  // Chat History / Thread State
  const initialData = useRef(loadInitialThreads()).current;
  const [threads, setThreads] = useState<ChatThread[]>(initialData.threads);
  const [activeThreadId, setActiveThreadId] = useState<string>(initialData.activeId);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(true);

  // Active Thread Reference for Async Callbacks
  const activeThreadIdRef = useRef<string>(activeThreadId);
  useEffect(() => {
    activeThreadIdRef.current = activeThreadId;
  }, [activeThreadId]);

  // Persist Threads to LocalStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_THREADS_KEY, JSON.stringify(threads));
    } catch (err) {
      console.error('Failed to save threads to localStorage', err);
    }
  }, [threads]);

  // Persist Active Thread ID to LocalStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_ACTIVE_KEY, activeThreadId);
    } catch (err) {
      console.error('Failed to save activeThreadId to localStorage', err);
    }
  }, [activeThreadId]);

  // Active Messages
  const activeThread = threads.find((t) => t.id === activeThreadId) || threads[0] || createNewThread();
  const messages = activeThread.messages;

  // Connection & UI State
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [status, setStatus] = useState('Disconnected');
  const [error, setError] = useState<string | null>(null);
  const [textInput, setTextInput] = useState('');

  // Voice Mode Flag: Only play incoming audio aloud when user interacted via voice
  const isVoiceModeRef = useRef<boolean>(false);

  // Per-turn flag: set when a clinical_answer has already been rendered for the current turn.
  // Prevents the Gemini Live spoken-text summary from creating a second assistant message
  // for text-mode queries (where the clinical card is already the complete answer).
  const hasClinicalAnswerThisTurnRef = useRef<boolean>(false);

  // Concurrency & Cancellation Controls
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const currentRequestIdRef = useRef<number>(0);
  const activeRequestThreadIdRef = useRef<string | null>(null);
  const isRequestCancelledRef = useRef<boolean>(false);
  const requestStartTimeRef = useRef<number>(0);

  // References
  const socketRef = useRef<WebSocket | null>(null);
  const recordingCtxRef = useRef<AudioContext | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef<number>(0);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-scroll to bottom of conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking]);

  // Connect on initial load
  useEffect(() => {
    connect();
    return () => {
      socketRef.current?.close();
    };
  }, []);

  // --- Voice Feature: Preserved Implementation ---
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

  const stopAudioPlayback = () => {
    // Immediately halt all currently active audio buffer sources
    activeSourcesRef.current.forEach((source) => {
      try {
        source.stop();
        source.disconnect();
      } catch {
        // Source might already have ended
      }
    });
    activeSourcesRef.current.clear();

    nextPlayTimeRef.current = 0;
    if (playbackCtxRef.current && playbackCtxRef.current.state === 'running') {
      nextPlayTimeRef.current = playbackCtxRef.current.currentTime;
    }
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
          const targetId = activeRequestThreadIdRef.current || activeThreadIdRef.current;

          if (msg.type === 'user_transcription') {
            if (isRequestCancelledRef.current) return;
            // Ignore transcription echoes from Gemini Live for text-mode queries.
            // When the user types (not speaks), isVoiceModeRef is false and the user
            // message is already displayed optimistically by handleSend().
            if (!isVoiceModeRef.current) return;
            const transcriptText = msg.text?.trim();
            if (!transcriptText) return;

            isVoiceModeRef.current = true;

            setThreads((prevThreads) => {
              return prevThreads.map((thread) => {
                if (thread.id !== targetId) return thread;

                const msgs = [...thread.messages];
                const lastMsg = msgs[msgs.length - 1];

                if (lastMsg && lastMsg.role === 'user' && lastMsg.isTranscription) {
                  // Update in-progress transcription message in place
                  msgs[msgs.length - 1] = {
                    ...lastMsg,
                    text: transcriptText,
                    isPartial: !msg.is_final,
                    isVoiceInteraction: true,
                  };
                } else {
                  // Append new user transcription message
                  msgs.push({
                    id: 'msg_user_' + Date.now(),
                    role: 'user',
                    text: transcriptText,
                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                    isTranscription: true,
                    isPartial: !msg.is_final,
                    isVoiceInteraction: true,
                  });
                }

                // Generate title from first user query
                let newTitle = thread.title;
                if ((thread.title === 'New Consultation' || thread.title === 'New Chat') && msg.is_final) {
                  newTitle = transcriptText.length > 36 ? transcriptText.slice(0, 36) + '...' : transcriptText;
                }

                return {
                  ...thread,
                  title: newTitle,
                  updatedAt: new Date().toISOString(),
                  messages: msgs,
                };
              });
            });

            if (msg.is_final) {
              requestStartTimeRef.current = Date.now();
              setIsThinking(true);
            }

          } else if (msg.type === 'text') {
            if (isRequestCancelledRef.current) return;
            // In text mode, after a clinical_answer has already been rendered for this
            // turn, suppress the redundant Gemini Live spoken-text summary so there
            // is exactly one assistant message per user query.
            if (!isVoiceModeRef.current && hasClinicalAnswerThisTurnRef.current) return;
            setIsThinking(false);

            const durationSec = requestStartTimeRef.current
              ? ((Date.now() - requestStartTimeRef.current) / 1000).toFixed(1)
              : undefined;

            setThreads((prevThreads) => {
              return prevThreads.map((thread) => {
                if (thread.id !== targetId) return thread;

                const msgs = [...thread.messages];
                const lastMsg = msgs[msgs.length - 1];

                if (lastMsg && lastMsg.role === 'assistant' && !lastMsg.clinicalData) {
                  msgs[msgs.length - 1] = {
                    ...lastMsg,
                    text: lastMsg.text + msg.text,
                    durationSec: lastMsg.durationSec || durationSec,
                  };
                } else {
                  msgs.push({
                    id: 'msg_asst_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
                    role: 'assistant',
                    text: msg.text,
                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                    isVoiceInteraction: isVoiceModeRef.current,
                    durationSec,
                  });
                }

                return {
                  ...thread,
                  updatedAt: new Date().toISOString(),
                  messages: msgs,
                };
              });
            });

          } else if (msg.type === 'audio') {
            // Only play audio chunk if voice mode is active and request not cancelled
            playAudioChunk(msg.data);

          } else if (msg.type === 'clinical_answer') {
            if (isRequestCancelledRef.current) return;
            setIsThinking(false);
            const parsed = parseClinicalAnswer(msg.answer);

            // Mark that a clinical answer card has been rendered for this turn.
            // The 'text' handler will use this to avoid adding a second assistant message.
            hasClinicalAnswerThisTurnRef.current = true;

            const durationSec = requestStartTimeRef.current
              ? ((Date.now() - requestStartTimeRef.current) / 1000).toFixed(1)
              : undefined;

            setThreads((prevThreads) => {
              return prevThreads.map((thread) => {
                if (thread.id !== targetId) return thread;

                const msgs = [...thread.messages];
                msgs.push({
                  id: 'msg_asst_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
                  role: 'assistant',
                  text: parsed.answer,
                  clinicalData: parsed,
                  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                  isVoiceInteraction: isVoiceModeRef.current,
                  durationSec,
                });

                return {
                  ...thread,
                  updatedAt: new Date().toISOString(),
                  messages: msgs,
                };
              });
            });

          } else if (msg.type === 'usage_metadata') {
            if (isRequestCancelledRef.current) return;
            const totalTokens = msg.total_tokens;
            if (typeof totalTokens === 'number' && totalTokens > 0) {
              setThreads((prevThreads) => {
                return prevThreads.map((thread) => {
                  if (thread.id !== targetId) return thread;

                  const msgs = [...thread.messages];
                  for (let i = msgs.length - 1; i >= 0; i--) {
                    if (msgs[i].role === 'assistant') {
                      msgs[i] = {
                        ...msgs[i],
                        tokenCount: totalTokens,
                      };
                      break;
                    }
                  }

                  return {
                    ...thread,
                    messages: msgs,
                  };
                });
              });
            }

          } else if (msg.type === 'interrupted') {
            console.log('Turn interrupted by user speech');
            stopAudioPlayback();
            setIsThinking(false);
            hasClinicalAnswerThisTurnRef.current = false;

          } else if (msg.type === 'turn_complete') {
            setIsThinking(false);
            hasClinicalAnswerThisTurnRef.current = false;

          } else if (msg.type === 'error') {
            setIsThinking(false);
            setError(msg.message);
          }
        } catch (e) {
          console.error('Failed to parse incoming WebSocket message', e);
        }
      };

      socket.onclose = () => {
        stopRecording();
        stopAudioPlayback();
        setIsConnected(false);
        setIsRecording(false);
        setIsThinking(false);
        setStatus('Disconnected');
      };

      socket.onerror = (err) => {
        stopRecording();
        stopAudioPlayback();
        setIsThinking(false);
        setIsRecording(false);
        setError('WebSocket error occurred');
        console.error('WebSocket error:', err);
      };

    } catch (e) {
      stopRecording();
      stopAudioPlayback();
      setIsThinking(false);
      setIsRecording(false);
      setError('Failed to connect to server');
    }
  };

  const disconnect = () => {
    socketRef.current?.close();
    stopRecording();
    stopAudioPlayback();
    setIsThinking(false);
    if (playbackCtxRef.current && playbackCtxRef.current.state !== 'closed') {
      playbackCtxRef.current.close();
      playbackCtxRef.current = null;
    }
  };

  const startRecording = async () => {
    // Prevent voice recording if text query or other request is currently active
    if (isThinking) {
      return;
    }

    if (!isConnected || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to assistant. Reconnecting...');
      connect();
      return;
    }

    try {
      isVoiceModeRef.current = true;
      isRequestCancelledRef.current = false;
      hasClinicalAnswerThisTurnRef.current = false;
      currentRequestIdRef.current++;
      activeRequestThreadIdRef.current = activeThreadIdRef.current;

      // Warm up playback context on user gesture
      const pCtx = getPlaybackContext();
      if (pCtx.state === 'suspended') {
        await pCtx.resume();
      }

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

  const handleStopVoice = () => {
    isRequestCancelledRef.current = true;
    currentRequestIdRef.current++;
    stopRecording();
    stopAudioPlayback();
    setIsThinking(false);

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'stop' }));
    }
  };

  const handleStopText = () => {
    isRequestCancelledRef.current = true;
    currentRequestIdRef.current++;
    setIsThinking(false);
    stopAudioPlayback();

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'stop' }));
    }
  };

  const playAudioChunk = async (base64Audio: string) => {
    // Only play audio chunk aloud if the user interacted via voice mode and request is active
    if (!isVoiceModeRef.current || isRequestCancelledRef.current) {
      return;
    }

    try {
      const audioCtx = getPlaybackContext();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

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

      activeSourcesRef.current.add(source);
      source.onended = () => {
        activeSourcesRef.current.delete(source);
      };

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
  // --- End of Voice Feature ---

  // Chat Actions
  const handleSend = (textToSend?: string) => {
    // Mutual Exclusion: block text query if voice mode or request is busy
    if (isThinking || isRecording) {
      return;
    }

    const message = (textToSend !== undefined ? textToSend : textInput).trim();
    if (!message) return;

    if (!isConnected || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to assistant. Reconnecting...');
      connect();
      return;
    }

    // Explicitly in text mode: do not play audio aloud unexpectedly
    isVoiceModeRef.current = false;
    stopAudioPlayback();

    // Reset cancellation & assign request sequence
    isRequestCancelledRef.current = false;
    hasClinicalAnswerThisTurnRef.current = false;
    currentRequestIdRef.current++;
    const targetId = activeThreadIdRef.current;
    activeRequestThreadIdRef.current = targetId;
    requestStartTimeRef.current = Date.now();

    socketRef.current.send(JSON.stringify({
      type: 'text',
      text: message,
    }));

    setThreads((prevThreads) => {
      return prevThreads.map((thread) => {
        if (thread.id !== targetId) return thread;

        const msgs = [...thread.messages];
        msgs.push({
          id: 'msg_user_' + Date.now(),
          role: 'user',
          text: message,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isVoiceInteraction: false,
        });

        let newTitle = thread.title;
        if (thread.title === 'New Consultation' || thread.title === 'New Chat') {
          newTitle = message.length > 36 ? message.slice(0, 36) + '...' : message;
        }

        return {
          ...thread,
          title: newTitle,
          updatedAt: new Date().toISOString(),
          messages: msgs,
        };
      });
    });

    setTextInput('');
    setIsThinking(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isThinking && !isRecording && textInput.trim()) {
        handleSend();
      }
    }
  };

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setTextInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
  };

  const handleNewChat = () => {
    isRequestCancelledRef.current = true;
    currentRequestIdRef.current++;
    if (isRecording) {
      stopRecording();
    }
    stopAudioPlayback();
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'stop' }));
    }
    isVoiceModeRef.current = false;
    const newThread = createNewThread();
    setThreads((prev) => [newThread, ...prev]);
    setActiveThreadId(newThread.id);
    activeRequestThreadIdRef.current = newThread.id;
    setIsThinking(false);
    setError(null);
    setTextInput('');
  };

  const handleSelectThread = (id: string) => {
    if (id === activeThreadId) return;
    isRequestCancelledRef.current = true;
    currentRequestIdRef.current++;
    if (isRecording) {
      stopRecording();
    }
    stopAudioPlayback();
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: 'stop' }));
    }
    isVoiceModeRef.current = false;
    setActiveThreadId(id);
    activeRequestThreadIdRef.current = id;
    setIsThinking(false);
    setError(null);
  };

  const handleDeleteThread = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (id === activeThreadId) {
      isRequestCancelledRef.current = true;
      currentRequestIdRef.current++;
      if (isRecording) {
        stopRecording();
      }
      stopAudioPlayback();
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({ type: 'stop' }));
      }
    }
    stopAudioPlayback();
    setThreads((prev) => {
      const filtered = prev.filter((t) => t.id !== id);
      if (filtered.length === 0) {
        const fresh = createNewThread();
        setActiveThreadId(fresh.id);
        activeRequestThreadIdRef.current = fresh.id;
        return [fresh];
      }
      if (id === activeThreadId) {
        setActiveThreadId(filtered[0].id);
        activeRequestThreadIdRef.current = filtered[0].id;
      }
      return filtered;
    });
  };

  const normalizedStatusClass = status.toLowerCase().replace(/[^a-z]/g, '');

  return (
    <div className="chat-app-root">
      {/* ChatGPT-Style Collapsible Sidebar */}
      <aside className={`chat-sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <button onClick={handleNewChat} className="btn-new-chat">
            <span className="btn-new-chat-icon">➕</span>
            <span className="btn-new-chat-label">New Consultation</span>
          </button>
        </div>

        <div className="sidebar-history-container">
          <div className="sidebar-section-title">Previous Consultations</div>
          <div className="sidebar-threads-list">
            {threads.map((thread) => (
              <div
                key={thread.id}
                className={`thread-item ${thread.id === activeThreadId ? 'active' : ''}`}
                onClick={() => handleSelectThread(thread.id)}
                title={thread.title}
              >
                <span className="thread-item-icon">💬</span>
                <span className="thread-item-title">{thread.title}</span>
                <button
                  className="thread-item-delete"
                  onClick={(e) => handleDeleteThread(e, thread.id)}
                  title="Delete Conversation"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-footer-text">⚕️ Indian OPD Guidelines</div>
          <div className="sidebar-footer-sub">Persistent Reference Storage</div>
        </div>
      </aside>

      {/* Main Chat Workspace */}
      <div className="chat-main-wrapper">
        {/* Top Navigation Bar */}
        <header className="chat-topbar">
          <div className="topbar-left">
            <button
              onClick={() => setIsSidebarOpen((prev) => !prev)}
              className="btn-sidebar-toggle"
              title={isSidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
            >
              ☰
            </button>
            <div className="topbar-brand">
              <span className="brand-icon">⚕️</span>
              <div>
                <h1 className="brand-title">{activeThread.title}</h1>
                <span className="brand-subtitle">Indian OPD Guideline Reference</span>
              </div>
            </div>
          </div>

          <div className="topbar-actions">
            <div className={`status-pill ${normalizedStatusClass}`}>
              <span className="status-dot"></span>
              {status}
            </div>

            {!isConnected ? (
              <button onClick={connect} className="btn-topbar-action">Connect</button>
            ) : (
              <button onClick={disconnect} className="btn-topbar-subtle">Disconnect</button>
            )}
          </div>
        </header>

        {/* Main Conversation Scroll Stream */}
        <main className="chat-scroll-container">
          {error && (
            <div className="chat-error-banner">
              <span>⚠️ {error}</span>
              <button onClick={() => setError(null)} className="error-close-btn">✕</button>
            </div>
          )}

          {messages.length === 0 ? (
            /* Welcome Screen when no conversation started */
            <div className="chat-welcome-container">
              <div className="welcome-hero-icon">⚕️</div>
              <h2 className="welcome-title">How can I assist your clinical practice?</h2>
              <p className="welcome-description">
                Ask medical reference questions grounded in Indian Outpatient Department (OPD) clinical guidelines.
                Speak into the microphone or type below.
              </p>

              <div className="starter-prompts-grid">
                {STARTER_PROMPTS.map((prompt, idx) => (
                  <button
                    key={idx}
                    className="starter-prompt-card"
                    onClick={() => handleSend(prompt)}
                  >
                    <span className="starter-prompt-text">{prompt}</span>
                    <span className="starter-prompt-arrow">→</span>
                  </button>
                ))}
              </div>

              <div className="welcome-disclaimer-pill">
                For clinician reference only • Grounded in reviewed guidelines • Not an autonomous diagnostic system
              </div>
            </div>
          ) : (
            /* Conversation Thread */
            <div className="chat-messages-thread">
              {messages.map((msg) => (
                <div key={msg.id} className={`chat-message-row ${msg.role} ${msg.isVoiceInteraction ? 'voice-mode' : ''}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? '👤' : '⚕️'}
                  </div>

                  <div className="message-bubble-wrapper">
                    <div className={`message-bubble ${msg.role} ${msg.isVoiceInteraction ? 'voice-mode' : ''}`}>
                      {/* Spoken Query Indicator */}
                      {msg.role === 'user' && msg.isTranscription && (
                        <div className="transcription-indicator">
                          <span className="transcription-indicator-badge">
                            <span className="transcription-mic-icon">🎙️</span>
                            <span>{msg.isPartial ? 'Speaking...' : 'Spoken Query'}</span>
                          </span>
                        </div>
                      )}

                      {/* Main Text Content */}
                      <div className={`message-text ${msg.isVoiceInteraction ? 'voice-transcript-text' : ''}`}>
                        {renderFormattedText(msg.text)}
                      </div>

                      {/* Integrated Clinical Guidance Metadata (Citations, Caution, Review Required) */}
                      {msg.clinicalData && (
                        <div className="clinical-metadata-card">
                          {msg.clinicalData.clinicianReviewRequired && (
                            <div className="clinical-review-pill">
                              <span className="pill-icon">⚠️</span>
                              <span>Clinician Review Required</span>
                            </div>
                          )}

                          {msg.clinicalData.caution && (
                            <div className="clinical-caution-box">
                              <span className="caution-icon">ℹ️</span>
                              <span className="caution-text">{msg.clinicalData.caution}</span>
                            </div>
                          )}

                          {msg.clinicalData.sources && msg.clinicalData.sources.length > 0 && (
                            <details className="clinical-sources-details">
                              <summary className="clinical-sources-summary">
                                <span className="summary-title">
                                  📚 {msg.clinicalData.sources.length} Cited Guideline Sources
                                </span>
                                <span className="summary-toggle">View Details</span>
                              </summary>
                              <div className="sources-tags-container">
                                {msg.clinicalData.sources.map((src, sIdx) => (
                                  <span key={sIdx} className="source-citation-badge">
                                    {src}
                                  </span>
                                ))}
                              </div>
                            </details>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Metadata: Timestamp, AI Response Time, Token Usage */}
                    <div className="message-metadata">
                      <span className="message-timestamp">{msg.timestamp}</span>
                      {msg.role === 'assistant' && msg.durationSec && (
                        <span className="message-duration"> · {msg.durationSec} sec</span>
                      )}
                      {msg.role === 'assistant' && typeof msg.tokenCount === 'number' && msg.tokenCount > 0 && (
                        <span className="message-tokens"> · {msg.tokenCount.toLocaleString()} tokens</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {/* Thinking / Searching Indicator */}
              {isThinking && (
                <div className="chat-message-row assistant thinking">
                  <div className="message-avatar">⚕️</div>
                  <div className="thinking-bubble">
                    <div className="thinking-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <span className="thinking-label">Retrieving guideline evidence & reasoning...</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        {/* Fixed Bottom Input Area */}
        <footer className="chat-bottom-dock">
          <div className="chat-input-bar">
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              placeholder={
                !isConnected
                  ? "Connecting to assistant..."
                  : isRecording
                  ? "Listening... (Finish speaking or press Stop Voice)"
                  : isThinking
                  ? "Assistant is generating response..."
                  : "Ask a clinical question (Enter to send, Shift+Enter for newline)..."
              }
              value={textInput}
              onChange={handleTextareaInput}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={!isConnected || isRecording || isThinking}
            />

            <div className="input-action-buttons">
              {/* Preserved Voice Button: Enhanced for Mutual Exclusion */}
              <button
                type="button"
                onClick={isRecording ? handleStopVoice : startRecording}
                className={`btn-voice ${isRecording ? 'recording' : ''}`}
                title={
                  isThinking
                    ? 'Voice disabled while query is processing'
                    : isRecording
                    ? 'Stop Voice Input'
                    : 'Start Voice Input'
                }
                disabled={!isConnected || isThinking}
              >
                {isRecording ? (
                  <>
                    <span className="recording-dot"></span>
                    <span>Stop</span>
                  </>
                ) : (
                  <>
                    <span>🎙️</span>
                    <span className="btn-voice-text">Voice</span>
                  </>
                )}
              </button>

              {/* Send Button or Stop Button */}
              {isThinking ? (
                <button
                  type="button"
                  onClick={handleStopText}
                  className="btn-stop"
                  title="Stop Response"
                >
                  <span className="stop-square"></span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => handleSend()}
                  className="btn-send"
                  disabled={!textInput.trim() || !isConnected || isRecording}
                  title={isRecording ? "Voice interaction active" : "Send Message"}
                >
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                </button>
              )}
            </div>
          </div>

          <div className="chat-disclaimer-footer">
            Outputs are grounded in reviewed Indian OPD guidelines for clinician reference. Always exercise professional clinical judgment.
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;
