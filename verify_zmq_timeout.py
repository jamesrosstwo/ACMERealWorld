import time
import zmq
import sys
from unittest.mock import MagicMock

# Mock dependencies to test GripperInterface in isolation
sys.modules["polymetis_pb2"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["omegaconf"] = MagicMock()
sys.modules["polymetis"] = MagicMock()
sys.modules["paramiko"] = MagicMock()

sys.path.append('/home/james/ACMERealWorld')
from client.nuc import GripperInterface

def test_hang():
    print("Initializing gripper on non-existent port...")
    # Port 5559 is likely not open
    gripper = GripperInterface("127.0.0.1", port=5559)
    
    print("Attempting to get state (should timeout in ~2s)...")
    start_time = time.time()
    try:
        gripper.get_state()
    except KeyError:
        print("Caught expected KeyError (due to empty response on timeout).")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")
    
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f}s")
    
    if 1.5 < elapsed < 2.5:
        print("SUCCESS: Timeout worked as expected.")
    else:
        print("FAILURE: Time elapsed is outside expected range (2s).")

if __name__ == "__main__":
    test_hang()
