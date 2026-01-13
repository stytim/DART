import socket
import json
import time

HOST = '127.0.0.1'
PORT = 8080

def main():
    while True:
        try:
            print(f"Connecting to {HOST}:{PORT}...")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, PORT))
            print("Connected!")
            
            buffer = ""
            while True:
                data = s.recv(4096)
                if not data:
                    break
                buffer += data.decode('utf-8')
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line: continue
                    try:
                        frame = json.loads(line)
                        print(f"Received frame: Root Pos={frame.get('root_pos')}, Joints={len(frame.get('joints', []))}")
                    except json.JSONDecodeError:
                        print(f"JSON Error: {line}")
        except ConnectionRefusedError:
            print("Connection refused, retrying in 1s...")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
        finally:
            s.close()

if __name__ == "__main__":
    main()
