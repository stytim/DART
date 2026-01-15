"""
Unity Prompt Receiver - receives text prompts from Unity via TCP
"""
import socket
import json
import threading
import queue
from typing import Callable, Optional


class UnityPromptReceiver:
    """
    TCP server that receives text prompts from Unity.
    Runs on a separate port from the motion streamer (default: 8081).
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8081, 
                 on_prompt_received: Optional[Callable[[str], None]] = None):
        """
        Args:
            host: Host to bind to (0.0.0.0 for all interfaces)
            port: Port to listen on (default 8081, different from motion streaming)
            on_prompt_received: Optional callback function called when a prompt is received
        """
        self.host = host
        self.port = port
        self.on_prompt_received = on_prompt_received
        
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        self.prompt_queue = queue.Queue()
        self.lock = threading.Lock()
        
        self._start_server_thread()
    
    def _start_server_thread(self):
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()
    
    def _server_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"[UnityPromptReceiver] Listening for prompts on {self.host}:{self.port}...")
            self.is_running = True
            
            while self.is_running:
                try:
                    client, addr = self.server_socket.accept()
                    print(f"[UnityPromptReceiver] Unity client connected: {addr}")
                    
                    with self.lock:
                        if self.client_socket:
                            try:
                                self.client_socket.close()
                            except:
                                pass
                        self.client_socket = client
                    
                    # Handle this client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client, 
                        args=(client,), 
                        daemon=True
                    )
                    client_thread.start()
                    
                except OSError:
                    break
                    
        except Exception as e:
            print(f"[UnityPromptReceiver] Server error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def _handle_client(self, client_socket):
        """Handle incoming messages from a connected client"""
        buffer = ""
        
        try:
            while self.is_running:
                data = client_socket.recv(4096)
                if not data:
                    break
                
                buffer += data.decode('utf-8')
                
                # Process complete messages (newline-delimited JSON)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line:
                        self._process_message(line)
                        
        except (ConnectionResetError, BrokenPipeError):
            print("[UnityPromptReceiver] Client disconnected")
        except Exception as e:
            print(f"[UnityPromptReceiver] Client error: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message: str):
        """Process a received JSON message"""
        try:
            data = json.loads(message)
            
            if 'prompt' in data:
                prompt = data['prompt'].strip()
                print(f"[UnityPromptReceiver] Received prompt: '{prompt}'")
                
                # Add to queue
                self.prompt_queue.put(prompt)
                
                # Call callback if set
                if self.on_prompt_received:
                    self.on_prompt_received(prompt)
                    
        except json.JSONDecodeError as e:
            print(f"[UnityPromptReceiver] Invalid JSON: {e}")
    
    def get_prompt(self, block: bool = False, timeout: float = None) -> Optional[str]:
        """
        Get the next prompt from the queue.
        
        Args:
            block: If True, wait for a prompt. If False, return None if no prompt available.
            timeout: Maximum time to wait (only used if block=True)
            
        Returns:
            The prompt string, or None if no prompt available (when block=False)
        """
        try:
            return self.prompt_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def has_prompt(self) -> bool:
        """Check if there's a prompt available without removing it"""
        return not self.prompt_queue.empty()
    
    def clear_prompts(self):
        """Clear all pending prompts"""
        while not self.prompt_queue.empty():
            try:
                self.prompt_queue.get_nowait()
            except queue.Empty:
                break
    
    def send_response(self, response: dict):
        """Send a response back to Unity (optional)"""
        with self.lock:
            if self.client_socket:
                try:
                    json_data = json.dumps(response) + "\n"
                    self.client_socket.sendall(json_data.encode('utf-8'))
                except Exception as e:
                    print(f"[UnityPromptReceiver] Send error: {e}")
    
    def close(self):
        """Stop the server and close all connections"""
        self.is_running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
                
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        
        print("[UnityPromptReceiver] Server closed")


# Example usage
if __name__ == '__main__':
    def on_prompt(prompt):
        print(f"Callback received: {prompt}")
    
    receiver = UnityPromptReceiver(on_prompt_received=on_prompt)
    
    print("Waiting for prompts... (Ctrl+C to exit)")
    try:
        while True:
            prompt = receiver.get_prompt(block=True, timeout=1.0)
            if prompt:
                print(f"Got prompt from queue: {prompt}")
    except KeyboardInterrupt:
        print("\nShutting down...")
        receiver.close()
