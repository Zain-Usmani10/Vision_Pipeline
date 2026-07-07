import cv2

class StreamGatekeeper:
    def __init__(self, video_source, frame_queue, skip_rate=3):
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.skip_rate = skip_rate
        self.frame_count = 0

    def stream_loop(self):
        cap = cv2.VideoCapture(self.video_source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            if self.frame_count % self.skip_rate == 0:
                self.frame_queue.put((frame, self.frame_count))
            self.frame_count += 1
        
        cap.release()