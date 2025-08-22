from datetime import datetime
import socket, os
print(f"Welcome {os.environ.get('USER')} from {socket.gethostname()} at {datetime.now().isoformat(timespec='seconds')}")
