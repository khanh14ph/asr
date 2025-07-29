import asyncio
import websockets
import json
import numpy as np
import base64
import sounddevice as sd
import threading
import time
from queue import Queue
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASRClient:
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.audio_queue = Queue()
        self.is_recording = False
        self.sample_rate = 16000  # Match server sample rate
        
        # Tính toán chunk size dựa trên model requirements
        # Emformer model thường cần ~320ms segments
        self.chunk_duration = 0.32  # 320ms
        self.chunk_size = int(self.sample_rate * self.chunk_duration)  # 5120 samples
        
        logger.info(f"Audio settings: {self.sample_rate}Hz, chunk size: {self.chunk_size} samples ({self.chunk_duration}s)")
        
    async def connect(self):
        """Connect to the ASR server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            logger.info(f"Connected to ASR server at {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from server")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if self.is_recording:
            # Convert to mono if stereo
            if len(indata.shape) > 1:
                audio_data = indata[:, 0]
            else:
                audio_data = indata.flatten()
            
            # Ensure we have the right data type
            audio_data = audio_data.astype(np.float32)
            self.audio_queue.put(audio_data.copy())
    
    def start_recording(self):
        """Start audio recording"""
        self.is_recording = True
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            self.stream.start()
            logger.info(f"Started recording with chunk size: {self.chunk_size} samples")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        logger.info("Stopped recording")
    
    async def send_audio_data(self, audio_data):
        """Send audio data to server"""
        if not self.websocket:
            return
        
        try:
            # Ensure correct data type and shape
            audio_data = audio_data.astype(np.float32)
            logger.debug(f"Sending audio chunk: {len(audio_data)} samples")
            
            # Encode audio data as base64
            audio_bytes = audio_data.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            message = {
                'type': 'audio',
                'audio': audio_b64,
                'timestamp': time.time()
            }
            
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending audio data: {e}")
    
    async def listen_for_responses(self):
        """Listen for responses from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if data['type'] == 'transcript':
                    print(f"\rTranscript: {data['text']}", end='', flush=True)
                elif data['type'] == 'reset_ack':
                    logger.info("Session reset acknowledged")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to server closed")
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
    
    async def reset_session(self):
        """Reset the ASR session on server"""
        if self.websocket:
            message = {'type': 'reset'}
            await self.websocket.send(json.dumps(message))
    
    async def run_streaming(self):
        """Main streaming loop"""
        if not await self.connect():
            return
        
        # Start listening for responses
        response_task = asyncio.create_task(self.listen_for_responses())
        
        # Start recording
        self.start_recording()
        
        try:
            print("Streaming started. Press Ctrl+C to stop...")
            while True:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    await self.send_audio_data(audio_data)
                else:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop_recording()
            response_task.cancel()
            await self.disconnect()

async def main():
    client = ASRClient()
    await client.run_streaming()

if __name__ == "__main__":
    asyncio.run(main())
