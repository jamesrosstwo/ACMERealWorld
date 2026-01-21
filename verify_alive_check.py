import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules["polymetis_pb2"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["omegaconf"] = MagicMock()
sys.modules["polymetis"] = MagicMock()
sys.modules["paramiko"] = MagicMock()

sys.path.append('/home/james/ACMERealWorld')
from client.nuc import GripperInterface, GripperState

def test_alive_check():
    print("Test 1: Connection Timeout/Failure")
    gripper_fail = GripperInterface("127.0.0.1", port=5559)
    # Mock zmq socket behaviors
    # By default, real socket will timeout. 
    # But since we are importing the real class, it uses real ZMQ.
    # The previous test confirmed timeout works.
    # So test_connection() should catch the error/timeout and return False.
    
    alive = gripper_fail.test_connection()
    print(f"Alive check result (should be False): {alive}")
    
    if not alive:
        print("SUCCESS: correctly reported dead.")
    else:
        print("FAILURE: reported alive despite no server.")

    print("\nTest 2: Connection Success (Mocking the socket internal)")
    gripper_success = GripperInterface("127.0.0.1", port=5558)
    
    # Mocking the _send_command to return a valid dict
    # We can't easily mock the internal socket of an existing instance without looking up the attribute name.
    # But we can mock `_send_command` itself since it's a method on the class instance?
    # No, that's messy. Let's mock the socket's recv_json.
    
    mock_socket = MagicMock()
    mock_socket.recv_json.return_value = {'width': 0.05, 'max_width': 0.1, 'is_grasped': False}
    gripper_success._socket = mock_socket
    
    alive_success = gripper_success.test_connection()
    print(f"Alive check result (should be True): {alive_success}")
    
    if alive_success:
        print("SUCCESS: correctly reported alive.")
    else:
        print("FAILURE: reported dead despite mock success.")

if __name__ == "__main__":
    test_alive_check()
