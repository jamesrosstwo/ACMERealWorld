import threading
import queue
import numpy as np
from tqdm import tqdm
from client.record import start_pipeline, enumerate_devices, get_tmstmp


class RealSenseInterface:
    class _FrameGrabberThread(threading.Thread):
        def __init__(self, idx, pipe, align, out_queue, stop_event):
            super().__init__()
            self.idx = idx
            self.pipe = pipe
            self.align = align
            self.out_queue = out_queue
            self.stop_event = stop_event

        def run(self):
            while not self.stop_event.is_set():
                try:
                    fs = self.pipe.wait_for_frames(timeout_ms=5000)
                    fs = self.align.process(fs) if self.align else fs
                    depth_frame = fs.get_depth_frame()
                    color_frame = fs.get_color_frame()

                    depth = np.asanyarray(depth_frame.get_data())
                    color = np.asanyarray(color_frame.get_data())

                    depth_ts = get_tmstmp(depth_frame)
                    color_ts = get_tmstmp(color_frame)

                    self.out_queue.put(((color, color_ts), (depth, depth_ts)))
                except Exception as e:
                    print(f"Camera {self.idx} failed to grab frame: {e}")
                    self.out_queue.put(((None, None), (None, None)))

    def __init__(self, width, height, fps):
        self._width = width
        self._height = height
        self._fps = fps
        self._pipelines, self._aligners = self._initialize_cameras()
        self._frame_queues = [queue.Queue() for _ in self._pipelines]
        self._stop_event = threading.Event()
        self._threads = []

        for idx, (pipe, align) in enumerate(zip(self._pipelines, self._aligners)):
            t = self._FrameGrabberThread(idx, pipe, align, self._frame_queues[idx], self._stop_event)
            t.start()
            self._threads.append(t)

    def _initialize_cameras(self):
        cameras = enumerate_devices()
        if not cameras:
            print("No RealSense devices detected – exiting.")
            return [], []

        print(f"Found {len(cameras)} camera(s):")
        for idx, (serial, product) in enumerate(cameras):
            print(f"   Camera {idx}: {serial}  ({product})")

        pipelines = []
        aligners = []
        for serial, _ in cameras:
            pipe, align, _ = start_pipeline(serial, self._width, self._height, self._fps)
            pipelines.append(pipe)
            aligners.append(align)
        return pipelines, aligners

    def get_synchronized_frames(self, n):
        for _ in tqdm(range(n)):
            frame_data = [q.get() for q in self._frame_queues]
            colors = [fd[0] for fd in frame_data]
            depths = [fd[1] for fd in frame_data]
            yield colors, depths

    def get_synchronized_frame(self):
        frame_data = [q.get() for q in self._frame_queues]
        colors = [fd[0] for fd in frame_data]
        depths = [fd[1] for fd in frame_data]
        return colors, depths

    def shutdown(self):
        self._stop_event.set()
        for t in self._threads:
            t.join()
        print("All camera threads stopped.")