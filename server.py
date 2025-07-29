import asyncio
import websockets
import json
import torch
import torchaudio
import base64
import numpy as np
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASRServer:
    def __init__(self):
        # Load model
        self.bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
        self.feature_extractor = self.bundle.get_streaming_feature_extractor()
        self.decoder = self.bundle.get_decoder()
        self.token_processor = self.bundle.get_token_processor()
        self.sample_rate = self.bundle.sample_rate
        self.segment_length = self.bundle.segment_length * self.bundle.hop_length
        self.context_length = self.bundle.right_context_length * self.bundle.hop_length
        
        logger.info(f"Sample rate: {self.sample_rate}")
        logger.info(f"Segment length: {self.segment_length} frames")
        logger.info(f"Context length: {self.context_length} frames")
        logger.info(f"Expected segment duration: {self.segment_length / self.sample_rate:.3f} seconds")

class ContextCacher:
    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        # Đảm bảo chunk có đúng kích thước segment_length
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        elif chunk.size(0) > self.segment_length:
            # Nếu chunk lớn hơn, chỉ lấy phần đầu
            chunk = chunk[:self.segment_length]
            
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :] if chunk.size(0) >= self.context_length else chunk
        return chunk_with_context

class ASRSession:
    def __init__(self, server: ASRServer):
        self.server = server
        self.cacher = ContextCacher(server.segment_length, server.context_length)
        self.state = None
        self.hypothesis = None
        self.audio_buffer = torch.tensor([])  # Buffer để tích lũy audio
        
    @torch.inference_mode()
    def process_audio_chunk(self, audio_data: np.ndarray) -> str:
        # Convert numpy array to torch tensor
        chunk = torch.from_numpy(audio_data).float()
        
        # Thêm vào buffer
        self.audio_buffer = torch.cat([self.audio_buffer, chunk])
        
        transcript = ""
        
        # Xử lý các segment đầy đủ từ buffer
        while self.audio_buffer.size(0) >= self.server.segment_length:
            # Lấy một segment từ buffer
            segment_data = self.audio_buffer[:self.server.segment_length]
            self.audio_buffer = self.audio_buffer[self.server.segment_length:]
            
            # Process the segment
            segment = self.cacher(segment_data)
            features, length = self.server.feature_extractor(segment)
            
            try:
                hypos, self.state = self.server.decoder.infer(
                    features, length, 10, state=self.state, hypothesis=self.hypothesis
                )
                self.hypothesis = hypos
                transcript = self.server.token_processor(hypos[0][0], lstrip=False)
            except Exception as e:
                logger.error(f"Error in decoder inference: {e}")
                logger.error(f"Features shape: {features.shape}, Length: {length}")
                continue
        
        return transcript

async def handle_client(websocket):
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    logger.info(f"Client {client_id} connected")
    
    # Create ASR session for this client
    asr_session = ASRSession(asr_server)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data['type'] == 'audio':
                    # Decode base64 audio data
                    audio_bytes = base64.b64decode(data['audio'])
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    logger.debug(f"Received audio chunk: {len(audio_array)} samples")
                    
                    # Process audio
                    transcript = asr_session.process_audio_chunk(audio_array)
                    
                    # Send response only if we have transcript
                    if transcript.strip():
                        response = {
                            'type': 'transcript',
                            'text': transcript,
                            'timestamp': data.get('timestamp', 0)
                        }
                        await websocket.send(json.dumps(response))
                    
                elif data['type'] == 'reset':
                    # Reset session
                    asr_session = ASRSession(asr_server)
                    await websocket.send(json.dumps({'type': 'reset_ack'}))
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                import traceback
                traceback.print_exc()
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error with client {client_id}: {e}")

# Global server instance
asr_server = ASRServer()

async def main():
    logger.info("Starting ASR WebSocket server on localhost:8765")
    server = await websockets.serve(handle_client, "localhost", 8765)
    logger.info("Server started successfully")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
