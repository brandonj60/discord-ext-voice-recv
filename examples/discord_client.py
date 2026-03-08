import argparse
import asyncio
import queue
import sys
import time
import torch
import subprocess

import aiohttp
import numpy as np
import sphn, discord
from discordtest import listening
import sounddevice as sd
import io
import wave
from scipy import signal
import soundfile as sf
import glob, os
import re
import logging
import threading

from modules.client_utils import AnyPrinter, Printer, RawPrinter

TEXT_CHANNEL_ID = 1114587482026147890

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("discord.voice").setLevel(logging.DEBUG)
logging.getLogger("discord.gateway").setLevel(logging.INFO)
logging.getLogger("discord.ext.listening").setLevel(logging.DEBUG)

class PatchedListeningVoiceClient(listening.VoiceClient):
    def __init__(self, client, channel):
        super().__init__(client, channel)

        # If discord.py doesn't expose VoiceClient._connected but does have it on
        # its internal voice connection state, alias it.
        if not hasattr(self, "_connected"):
            conn = getattr(self, "_connection", None)
            ev = getattr(conn, "_connected", None)
            if ev is None:
                ev = threading.Event()
                ev.set()
            self._connected = ev

def convert_opus_to_pcm(pcm_data, input_rate=24000, output_rate=48000, output_type="int16"):
    # Resample the PCM data from 24kHz to 48kHz
    num_samples = int(len(pcm_data) * output_rate / input_rate)
    
    if output_type == "float":
        pcm_data = pcm_data.astype(np.float32) / 32768.0
        
    # Resample the float32 data
    resampled_pcm_data = signal.resample(pcm_data, num_samples)

    # Convert from float32 (-1.0 to 1.0) to int16 (-32768 to 32767)
    #resampled_int16_data = np.clip(resampled_pcm_data * 32767, -32768, 32767).astype(np.int16)
    if output_type != "float":
        resampled_pcm_data = (resampled_pcm_data * 32767).astype(np.int16)
        resampled_pcm_data = np.stack((resampled_pcm_data, resampled_pcm_data), axis=-1) #stereo
        
    return resampled_pcm_data
    
def cleanup_directory(directory, max_files=100):
    # Get list of all files in the directory sorted by modification time (oldest first)
    files = sorted(
        glob.glob(os.path.join(directory, "*")),
        key=os.path.getmtime
    )

    # Check if we have more than max_files
    if len(files) > max_files:
        files_to_delete = files[:len(files) - max_files]  # Get excess oldest files

        for file in files_to_delete:
            try:
                os.remove(file)
                print(f"Client Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

def restart():
    print("Restarting the application...")
    python = sys.executable
    os.execl(python, python, *sys.argv)  # Replaces the current process

class DiscordAudioHandler(discord.Client):
    def __init__(
        self,
        printer: AnyPrinter,
        uri,
        sample_rate: float = 24000,
        channels: int = 1,
        frame_size: int = 1920,
        intents = None
    ) -> None:
        super().__init__(intents=intents)
        self.printer = printer
        self.uri = uri
        self.websocket = None
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels
        self._done = False
        self._voice_done = False
        self._connected = asyncio.Event()
        self.listen_task = None
        self._process_pool = listening.AudioProcessPool(1)
        
        self._opus_writer = sphn.OpusStreamWriter(sample_rate)
        self._opus_reader = sphn.OpusStreamReader(sample_rate)
        self._send_queue = asyncio.Queue()
        
        self._session = None  # Store session for reuse
        self._tasks = []      # Track tasks for cancellation
            
    # Custom AudioSink to capture audio
    class AudioBufferSink(listening.AudioSink):
        """
        Receives decoded PCM frames from discord-ext-listening and pushes 24kHz mono float32
        into your sphn.OpusStreamWriter via _opus_writer.append_pcm(...).

        Notes:
        - discord-ext-listening's AudioFrame.audio is decoded PCM bytes (s16le).
        - Discord receive is typically 48kHz stereo s16le.
        """

        def __init__(self, _opus_writer, frame_size: int, channels: int):
            self._opus_writer = _opus_writer
            self.frame_size = frame_size
            self.channels = channels

            self._lock = threading.Lock()
            self._buffer = bytearray()

            # You were treating frame_size as "24k mono samples per chunk" (1920 @ 24kHz = 80ms)
            self._in_rate = 48000
            self._out_rate = 24000
            self._in_channels = 2               # discord receive: stereo
            self._bytes_per_sample = 2          # int16

            # How many input samples per channel correspond to frame_size output samples?
            # 1920 @ 24k -> 3840 @ 48k
            self._in_samples_per_chan = int(frame_size * self._in_rate / self._out_rate)  # e.g., 3840
            self._chunk_int16 = self._in_samples_per_chan * self._in_channels             # e.g., 7680
            self._chunk_bytes = self._chunk_int16 * self._bytes_per_sample                # e.g., 15360

            self._min_packet_bytes = 100  # keep your old "ignore tiny packets" behavior

        def on_audio(self, frame: listening.AudioFrame):
            try:
                audio_data = frame.audio
                # If you had a print after reading frame.audio, put it here:
                print("on_audio got bytes:", len(audio_data))

                if not audio_data or len(audio_data) < self._min_packet_bytes:
                    return

                chunks = []
                with self._lock:
                    self._buffer.extend(audio_data)
                    while len(self._buffer) >= self._chunk_bytes:
                        chunks.append(bytes(self._buffer[:self._chunk_bytes]))
                        del self._buffer[:self._chunk_bytes]

                # Process outside the lock
                for chunk in chunks:
                    in_data = np.frombuffer(chunk, dtype=np.int16)
                    if in_data.size < self._chunk_int16:
                        continue

                    stereo = in_data.reshape((self._in_samples_per_chan, self._in_channels))
                    mono = np.mean(stereo, axis=1).astype(np.int16)

                    # Your helper already does the right thing when output_type="float":
                    # returns float32 in [-1, 1] at 24kHz, length == frame_size (e.g., 1920)
                    in_data_float32 = convert_opus_to_pcm(
                        mono, input_rate=self._in_rate, output_rate=self._out_rate, output_type="float"
                    )

                    self._opus_writer.append_pcm(in_data_float32)

            except Exception as e:
                print(f"AudioBufferSink.on_audio error: {e}")

        def on_rtcp(self, packet):
            # You can ignore RTCP unless you need timing / SSRC events.
            return

        def cleanup(self):
            # Called when listening stops.
            with self._lock:
                self._buffer.clear()

    ### Read from the Queue of data and send to the server endpoint
    async def _queue_loop(self) -> None:
        while True:
            if self._done:  # Check if the handler is shutting down
                print("breaking _queue_loop")
                break
            try:
                msg_type, data = await self._send_queue.get()  # Get next message
                await self.websocket.send_bytes(bytes([msg_type]) + data)  # Send type + data
                print(f"Sent message type {msg_type} with {len(data)} bytes")
            except Exception as e:
                print(f"Error sending message: {e}")
                #self._lost_connection()  # Handle connection loss
                #return
            await asyncio.sleep(0.001)
        print("_queue_loop shutting down")

    async def _audio_send_loop(self) -> None:
        self.printer.log("info", f"Starting audio_send loop")
        silent_frame = np.zeros(1920, dtype=np.float32)  # 1920 samples at 24kHz = 80ms
        #fade_samples = int(1920 * 0.1)  # 10% fade-in for smooth silence
        #silent_frame[:fade_samples] *= np.linspace(0, 1, fade_samples)
        
        last_audio_time = time.time()
        frame_duration = 1920 / 24000  # ~80ms per frame
        silence_threshold = frame_duration   # *3? Inject silence after 240ms of no audio
        
        while True:
            if self._voice_done:
                return
            await asyncio.sleep(0.001)  # Small sleep to avoid busy-waiting
            msg = self._opus_writer.read_bytes()  # Read Opus-encoded audio
            if len(msg) > 0:
                await self._send_queue.put((1, msg))  # Queue audio with type 1
                last_audio_time = time.time()
            else:
                time_since_last_audio = time.time() - last_audio_time
                if time_since_last_audio > silence_threshold:
                    self._opus_writer.append_pcm(silent_frame)  # Inject silence
                    silent_msg = self._opus_writer.read_bytes()
                    #print(f"Silence at {time.time()}")
                    if len(silent_msg) > 0:
                        await self._send_queue.put((1, silent_msg))
                    last_audio_time = time.time()
                    
    ## Get data from Moshi and write to Discord pipe
    async def _decoder_loop(self, voice_client) -> None:
        self.printer.log("info", f"Starting decoder_loop")
        all_pcm_data = None
        pipe = None

        # Open a continuous pipe to FFmpeg and pass it to voice_client.play() once
        ffmpeg_command = [
            'ffmpeg',
            '-f', 's16le',  # Format: Signed 16-bit little-endian PCM data
            '-ar', '48000',  # Sample rate: 48kHz
            '-ac', '2',      # Number of channels: 2 (stereo)
            '-i', 'pipe:0',  # Input comes from stdin (pipe)
            '-f', 'wav',     # Output format is WAV
            'pipe:1'         # Output also goes to stdout (another pipe)
        ]

        # Open a subprocess for FFmpeg to handle the PCM to WAV conversion
        pipe = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        audio_source = discord.FFmpegPCMAudio(pipe.stdout, pipe=True)
        voice_client.play(audio_source)

        # Now feed PCM data continuously into FFmpeg's stdin (the pipe)
        while not self._voice_done and voice_client.is_connected():
            await asyncio.sleep(0.001)
            pcm = self._opus_reader.read_pcm()

            if all_pcm_data is None:
                all_pcm_data = pcm
            else:
                all_pcm_data = np.concatenate((all_pcm_data, pcm))

            # Process PCM data in chunks of frame_size
            chunk_duration = self.frame_size / 48000
            while all_pcm_data.shape[-1] >= self.frame_size:
                # Extract the chunk of size `frame_size`
                pcm_chunk = all_pcm_data[:self.frame_size]
                
                # Convert PCM chunk to int16 48kHz stereo data
                pcm_data_48k_int16 = convert_opus_to_pcm(pcm_chunk, input_rate=24000, output_rate=48000)
                try:
                    # Write the PCM data to the stdin of the FFmpeg process
                    pipe.stdin.write(pcm_data_48k_int16.tobytes())
                    await asyncio.sleep(chunk_duration)
                except:
                    print("Failed writing to pipe. Recreating")
                    pipe = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

                # Remove the processed chunk from the buffer
                all_pcm_data = all_pcm_data[self.frame_size:]

        # When done, make sure to close the FFmpeg process cleanly
        pipe.stdin.close()
        pipe.wait()

    async def _recv_loop(self) -> None:
        global TEXT_CHANNEL_ID
        while not self._done:
            await self._connected.wait()  # Waits for connection
            try:
                last_kind = 0
                async for message in self.websocket:
                    if message.type == aiohttp.WSMsgType.CLOSED:
                        self.printer.log("info", "Connection closed")
                        break
                    elif message.type == aiohttp.WSMsgType.ERROR:
                        self.printer.log("error", f"{self.websocket.exception()}")
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        self.printer.log("error", f"received from server: {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        self.printer.log(
                            "warning", f"unsupported message type {type(message)}"
                        )
                        continue
                    if len(message) == 0:
                        self.printer.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        self._opus_reader.append_bytes(payload)
                        print(f"Received data from websocket. time={time.time()}")
                    elif kind == 3:  # text
                        payload = message[1:]
                        self.printer.print_token(payload.decode())
                        channel = self.get_channel(TEXT_CHANNEL_ID)
                        if channel is None:
                            print("Channel not found. Make sure the bot is in the server and has access to the channel.")
                        else:
                            if not self.listen_task and last_kind == 3:
                                await asyncio.sleep(1.0)
                            await channel.send(payload.decode('utf-8', errors='ignore'))
                    else:
                        self.printer.log("warning", f"unknown message kind {kind}")
                    last_kind = kind
            except Exception as e:
                print(f"Error receiving from webSocket: {e}")
                #self._lost_connection()
                #return

    async def play_sound(self, filepath, vc, pipeAudio):
        if vc:
            while vc.is_playing():
                await asyncio.sleep(0.001)
        try:
            if vc is not None:
                vc.play(discord.PCMAudio(filepath))
        except discord.errors.ClientException as e:
            print("Error playing sound!")

        
    def _lost_connection(self) -> None:
        if not self._done:
            self.printer.log("error", "Lost connection with the server!")
            self._done = True

    async def on_message(self, message):
        global TEXT_CHANNEL_ID   # Replace with your channel ID (e.g., 1234567890)
        TEXT_CHANNEL_ID=message.channel.id
        if True or self.user.mentioned_in(message):   #channel.id == TEXT_CHANNEL_ID and 
            if message.author != self.user:
                for attachment in message.attachments:
                    if attachment.content_type.startswith('image/'):  # Check if it's an image
                        image_data = await attachment.read()  # Get image bytes
                        await self._send_queue.put((2, image_data))  # Queue with type 2
                        print(f"Queued image data of size {len(image_data)} bytes")
                if "restart" in message.content:
                    restart()
                elif "info" in message.content:
                    if isinstance(message.channel, discord.VoiceChannel):
                        embed = discord.Embed(title=f"Voice Channel: {channel.name}", color=discord.Color.blue())
                        embed.add_field(name="Bitrate", value=f"{channel.bitrate} bps", inline=False)
                        embed.add_field(name="User Limit", value=channel.user_limit if channel.user_limit else "No limit", inline=False)
                        embed.add_field(name="Region", value=str(channel.rtc_region) if channel.rtc_region else "Automatic", inline=False)
                        
                        await channel.send(embed=embed)
                elif "change character" in message.content:
                    print("Changing character")
                    name = re.search(r'change character (\w+)', message.content)
                    if name:
                        print(f"Sending character change: {name.group(1)}")
                        name = name.group(1).encode('utf-8')
                        await self._send_queue.put((4, name))
                elif "" in message.content: # and not message.attachments:
                    print(f"Text message received {message.content}")
                    user_bytes = message.author.display_name.encode('utf-8')
                    text_bytes = message.content.encode('utf-8')
                    combined_data = user_bytes + b'\x00' + text_bytes
                    await self._send_queue.put((3, combined_data))  # Single message


    async def on_voice_state_update(self, member, before, after):
        """ Most of the below logic is so the bot can follow users across channels """
        if member.id == self.user.id or "BrandonBot" in member.name:
            return

        # Detect join or switch
        if after.channel and (not self.voice_clients or before.channel != after.channel):
            print(f"{member.display_name} joined or switched to {after.channel.name}")

            # Cancel the previous listen task
            if self.listen_task and not self.listen_task.done():
                print("Cancelling previous listener task...")
                self._voice_done = True  # Signal shutdown to the loop
                self.listen_task.cancel()
                try:
                    await self.listen_task
                except asyncio.CancelledError:
                    pass

            # Disconnect old voice clients
            for vc in self.voice_clients:
                print("Disconnecting old voice clients")
                await vc.disconnect(force=True)

            # Reconnect to new voice channel
            print("Reconnect to new voice channel")

            try:
                await asyncio.sleep(1)
                vc = await after.channel.connect(cls=PatchedListeningVoiceClient,reconnect=False,self_deaf=False,self_mute=False)
                print("negotiated voice mode:", getattr(vc, "mode", None))
            except asyncio.TimeoutError:
                print("Failed to connect: timed out.")
            except ClientException as e:
                print(f"Client error: {e}")
            except OpusNotLoaded:
                print("Opus library not loaded. Cannot connect to voice.")
            if not vc.is_connected():
                print("Failed to connect. Retrying")
                vc = await after.channel.connect(cls=PatchedListeningVoiceClient,reconnect=False,self_deaf=False,self_mute=False)
            await self._send_queue.put((5, member.name.encode("utf-8")))
            self._voice_done = False  # Reset shutdown flag
            self.listen_task = asyncio.create_task(self.start_listening(vc))
            self.listen_task.add_done_callback(lambda t: print("listen_task exception:", t.exception()) if t.exception() else None)
            
        # Case 2: User disconnects completely
        elif before.channel and after.channel is None:
            print(f"{member.display_name} disconnected from {before.channel.name}")
            
            # Check if bot is still in a voice channel
            if self.voice_clients:
                vc = self.voice_clients[0]  # Current voice client
                if len(vc.channel.members) == 1:  # Only bot remains
                    print("Bot is alone in voice channel, shutting down listen task...")
                    if hasattr(self, 'listen_task') and self.listen_task and not self.listen_task.done():
                        self._voice_done = True
                        self.listen_task.cancel()
                        try:
                            await self.listen_task
                        except asyncio.CancelledError:
                            pass
                    try:
                        await vc.disconnect(force=True)  # Disconnect bot from voice channel
                    except:
                        print("Error encountered when exiting voice chat")
                    print("Bot disconnected from voice channel")
        
    async def on_listening_stopped(self, sink, exc=None):
        #sink.convert_files()  # convert whatever audio we have before throwing error

        # Raise any exceptions that may have occurred
        if exc is not None:
            raise exc
            
    async def start_listening(self, voice_client):
        while not voice_client.is_connected():
            print("Waiting for voice client to connect...")
            await asyncio.sleep(0.1)

        print("Starting to listen")

        sink = self.AudioBufferSink(self._opus_writer, self.frame_size, self.channels)

        voice_client.listen(
            sink,
            self._process_pool,
            after=self.on_listening_stopped
        )

        await asyncio.gather(
            self._decoder_loop(voice_client),
            self._audio_send_loop()
        )

    async def on_ready(self):
        print(f"Logged in as {self.user}")
        # Start non-voice loops
        self._session = aiohttp.ClientSession()
        self._tasks.extend([
            asyncio.create_task(self._connect_loop(self._session)),
            asyncio.create_task(self._recv_loop()),
            asyncio.create_task(self._queue_loop())
        ])

    async def _connect_loop(self, session):
        while not self._done:
            try:
                self.printer.log("info", f"Attempting to connect to {self.uri}")
                async with session.ws_connect(self.uri, timeout=aiohttp.ClientTimeout(total=10)) as ws:
                    self.websocket = ws
                    self._connected.set()
                    self.printer.log("info", "Connected successfully!")
                    await asyncio.sleep(float('inf'))
            except Exception as e:
                self.printer.log("error", f"Connection error: {e}. Retrying in 3 seconds...")
                self._connected.clear()
                await asyncio.sleep(3)

    async def close(self):
        self._done = True
        for task in self._tasks:
            task.cancel()
        if self._session:
            await self._session.close()
        try:
            if getattr(self, "_process_pool", None) is not None:
                self._process_pool.cleanup_processes()
        finally:
            await super().close()
    
async def runApp(printer: AnyPrinter, args, intents, token):
    if args.url is None:
        proto = "ws"
        if args.https:
            proto += "s"
        uri = f"{proto}://{args.host}:{args.port}/api/chat"
    else:
        proto = "wss"
        if '://' in args.url:
            proto, without_proto = args.url.split('://', 1)
            if proto in ['ws', 'http']:
                proto = "ws"
            elif proto in ['wss', 'https']:
                proto = "wss"
            else:
                printer.log("error", f"The provided URL {args.url} seems to contain a protocol but it is unknown.")
                sys.exit(1)
        else:
            without_proto = args.url
        uri = f"{proto}://{without_proto}/api/chat"

    printer.log("info", f"Attempting to connect to {uri}.")
    
    printer.log("info", "Connected successfully!")
    printer.print_header()
    connection = DiscordAudioHandler(printer, uri, intents=intents)

    await connection.start(token)  # This runs until disconnected


def main():
    parser = argparse.ArgumentParser("client_opus")
    parser.add_argument("--host", default="localhost", type=str, help="Hostname to connect to.")
    parser.add_argument("--port", default=8998, type=int, help="Port to connect to.")
    parser.add_argument("--https", action='store_true',
                        help="Set this flag for using a https connection.")
    parser.add_argument("--url", type=str, help='Provides directly a URL, e.g. to a gradio tunnel.')
    args = parser.parse_args()
    printer: AnyPrinter

    with open("makisebotkey.key", "r") as f:
        TOKEN = f.read()
    intents = discord.Intents.default()
    intents.voice_states = True
    intents.message_content = True
    
    if sys.stdout.isatty():
        printer = Printer()
    else:
        printer = RawPrinter()
    try:
        asyncio.run(runApp(printer, args, intents, TOKEN))
    except KeyboardInterrupt:
        printer.log("warning", "Interrupting, exiting connection.")

        print("cancelling tasks...")

        # Get all running tasks and cancel them
        loop = asyncio.get_event_loop()
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
    printer.log("info", "All done!")


if __name__ == "__main__":
    main()