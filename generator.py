import time, random, socket
from datetime import datetime

def generate_data():
    sensors = ["sensor_1", "sensor_2", "sensor_3", "sensor_4"]
    while True:
        sensor = random.choice(sensors)
        value = random.uniform(20, 100)
        timestamp = datetime.now().isoformat()
        yield f"{timestamp},{sensor},{value:.2f}\n"
        time.sleep(0.5)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('localhost', 9999))
server.listen(1)
print("Waiting for connection on port 9999...")
conn, addr = server.accept()
print(f"Connected to {addr}")

try:
    for data in generate_data():
        conn.send(data.encode())
        print(f"Sent: {data.strip()}")
except KeyboardInterrupt:
    print("\nGenerator stopped.")
finally:
    conn.close()
    server.close()