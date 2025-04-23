import cv2
import torch
import time
import math
import numpy as np
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class PlayerPerformanceAnalyzer:
    """
    Class to analyze player performance from video: detection, tracking, pose estimation,
    and calculation of speed, ball control, and self-pass metrics.
    """

    def __init__(self,
                 source_video_path: str,
                 output_video_path: str,
                 output_csv_path: str,
                 yolov_model_path: str,
                 confidence_threshold: float = 0.8,
                 control_threshold_m: float = 0.5,
                 px_to_m: float = 0.02):
        """
        Initialize video capture, writers, models, and parameters.
        Args:
            source_video_path: path to input video file
            output_video_path: path to save annotated video
            output_csv_path: path to save metrics CSV
            yolov_model_path: path to YOLOv9 model
            confidence_threshold: detection confidence threshold
            control_threshold_m: control radius in meters for self-pass detection
            px_to_m: pixel to meter conversion factor
        """
        # Video IO setup
        self.cap = cv2.VideoCapture(source_video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(output_video_path, fourcc, fps,
                                      (self.frame_width, self.frame_height))
        if not self.writer.isOpened():
            self.cap.release()
            raise IOError("Failed to open VideoWriter")

        # Models and trackers
        self.model = YOLO(yolov_model_path)
        self.tracker = DeepSort(max_age=50)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
                                          model_complexity=1,
                                          enable_segmentation=False,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Parameters
        self.CONF = confidence_threshold
        self.ALLOWED_CLASSES = [0, 32]  # 0=person, 32=sports ball
        self.control_thresh_m = control_threshold_m
        self.control_thresh_px = self.control_thresh_m / px_to_m
        self.px_to_m = px_to_m

        # Colors for drawing
        np.random.seed(42)
        self.colors = np.random.randint(0, 255,
                                        size=(len(self.model.names), 3))

        # Histories for metrics
        self.speed_history = {}
        self.keypoints_history = {}
        self.ball_history = []
        self.prev_positions = {}

        # Output
        self.output_csv_path = output_csv_path

    @staticmethod
    def draw_corner_rect(img, bbox, line_length=30,
                         line_thickness=5, rect_thickness=1,
                         rect_color=(255, 0, 255), line_color=(0, 255, 0)):
        """
        Draw a corner-style rectangle on an image.
        Args:
            img: image array
            bbox: tuple (x, y, w, h)
        Returns:
            annotated image
        """
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        if rect_thickness:
            cv2.rectangle(img, (x, y), (x2, y2), rect_color, rect_thickness)
        # draw corner lines
        # top-left
        cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
        cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)
        # top-right
        cv2.line(img, (x2, y), (x2 - line_length, y), line_color, line_thickness)
        cv2.line(img, (x2, y), (x2, y + line_length), line_color, line_thickness)
        # bottom-left
        cv2.line(img, (x, y2), (x + line_length, y2), line_color, line_thickness)
        cv2.line(img, (x, y2), (x, y2 - line_length), line_color, line_thickness)
        # bottom-right
        cv2.line(img, (x2, y2), (x2 - line_length, y2), line_color, line_thickness)
        cv2.line(img, (x2, y2), (x2, y2 - line_length), line_color, line_thickness)
        return img

    def assess_speed_quality(self, speeds_pps):
        """
        Convert pixel/sec speeds to m/s, compute average & max, classify quality.
        """
        valid_pps = [s for s in speeds_pps if s and s > 0]
        speeds_mps = [s * self.px_to_m for s in valid_pps] or [0.0]
        avg = float(np.mean(speeds_mps))
        mx = float(np.max(speeds_mps))
        if avg >= 6.0:
            quality = 'Excellent Speed'
            desc = 'Average speed >=6 m/s, outruns opponents effectively.'
        elif avg >= 4.0:
            quality = 'Good Speed'
            desc = 'Average speed between 4-6 m/s, solid game coverage.'
        else:
            quality = 'Poor Speed'
            desc = 'Average speed <4 m/s, may limit play pace.'
        return {'avg_mps': avg, 'max_mps': mx,
                'quality': quality, 'description': desc}

    def assess_ball_control_quality(self, player_id):
        """
        Evaluate average distance between ball and feet, classify control.
        """
        dists = []
        kps_hist = self.keypoints_history.get(player_id, [])
        n = min(len(kps_hist), len(self.ball_history))
        for i in range(n):
            kps = kps_hist[i]
            ball = self.ball_history[i]
            if ball is None or not kps:
                continue
            feet_pts = [kps[idx] for idx in (27, 28) if idx < len(kps)]
            if not feet_pts:
                continue
            d_px = min(math.hypot(ball[0]-x, ball[1]-y)
                       for x, y in feet_pts)
            dists.append(d_px * self.px_to_m)
        if not dists:
            return {'avg_dist': None,
                    'quality': 'Poor Control',
                    'description': 'Insufficient data.'}
        avg = float(np.mean(dists))
        if avg < 1.0:
            quality = 'Excellent Control'
            desc = 'Ball remains under 1 m, allows quick decisions.'
        elif avg < 2.0:
            quality = 'Good Control'
            desc = 'Ball 1-2 m away, still manageable.'
        else:
            quality = 'Poor Control'
            desc = 'Ball >2 m away, high loss risk.'
        return {'avg_dist': avg, 'quality': quality, 'description': desc}

    def assess_self_passes(self, player_id):
        """
        Detect self-pass events and classify by travel distance.
        """
        events = []
        in_pass = False
        start_idx = 0
        max_px = 0
        kps_hist = self.keypoints_history.get(player_id, [])
        n = min(len(kps_hist), len(self.ball_history))
        for i in range(n):
            kps = kps_hist[i]
            ball = self.ball_history[i]
            if ball is None or not kps:
                if in_pass:
                    in_pass, start_idx, max_px = False, 0, 0
                continue
            fx = np.mean([kps[idx][0] for idx in (27,28) if idx<len(kps)])
            fy = np.mean([kps[idx][1] for idx in (27,28) if idx<len(kps)])
            d_px = math.hypot(ball[0]-fx, ball[1]-fy)
            if not in_pass:
                if d_px > self.control_thresh_px:
                    in_pass = True
                    start_idx = i
                    max_px = d_px
            else:
                max_px = max(max_px, d_px)
                if d_px <= self.control_thresh_px:
                    dist_m = max_px * self.px_to_m
                    if 2 <= dist_m < 5:
                        q, d = 'Excellent Self Pass', '2-5 m, bypass opponents.'
                    elif 5 <= dist_m < 8:
                        q, d = 'Effective Self Pass', '5-8 m, no interception.'
                    else:
                        q, d = 'Poor Self Pass', '>8 m, risk of loss.'
                    events.append({'distance_m': dist_m,
                                   'quality': q,
                                   'description': d,
                                   'start_frame': start_idx,
                                   'end_frame': i})
                    in_pass, start_idx, max_px = False, 0, 0
        return events

    def export_metrics(self):
        """
        Aggregate all player metrics and save to CSV.
        """
        rows = []
        for pid, speeds in self.speed_history.items():
            speed_res = self.assess_speed_quality(speeds)
            ctrl_res = self.assess_ball_control_quality(pid)
            sp_events = self.assess_self_passes(pid)
            rows.append({
                'player_id': pid,
                'avg_speed_mps': speed_res['avg_mps'],
                'max_speed_mps': speed_res['max_mps'],
                'speed_quality': speed_res['quality'],
                'avg_control_distance_m': ctrl_res['avg_dist'],
                'control_quality': ctrl_res['quality'],
                'num_excellent_self_pass': sum(e['quality']=='Excellent Self Pass' for e in sp_events),
                'num_effective_self_pass': sum(e['quality']=='Effective Self Pass' for e in sp_events),
                'num_poor_self_pass': sum(e['quality']=='Poor Self Pass' for e in sp_events)
            })
        pd.DataFrame(rows).to_csv(self.output_csv_path, index=False)

    def process_video(self):
        """
        Main loop: detection, tracking, pose estimation, metric accumulation, and annotation.
        """
        frame_count = 0
        start_time = time.time()
        ball_id = 1
        last_ball_bbox = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detection
            results = self.model(frame)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            detections = []
            for (x1,y1,x2,y2), conf, cid in zip(boxes, confs, cls_ids):
                if conf < self.CONF or cid not in self.ALLOWED_CLASSES:
                    continue
                detections.append([[int(x1),int(y1),int(x2-x1),int(y2-y1)], float(conf), cid])

            # Person tracking
            persons = [d for d in detections if d[2]==0]
            tracks = self.tracker.update_tracks(persons, frame=frame)

            # Ball manual tracking
            balls = [d for d in detections if d[2]==32]
            if balls:
                best = max(balls, key=lambda x:x[1])
                last_ball_bbox = best[0]
                cx = best[0][0] + best[0][2]/2
                cy = best[0][1] + best[0][3]/2
            else:
                cx = cy = None
            self.ball_history.append((cx,cy) if cx else None)

            # Process each person track
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                x1,y1,x2,y2 = map(int, track.to_ltrb())
                roi = frame[y1:y2, x1:x2]
                if roi.size==0:
                    continue
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                res = self.pose.process(rgb)
                if not res.pose_landmarks:
                    continue
                # landmarks to pixel coords
                kps = []
                for lm in res.pose_landmarks.landmark:
                    kps.append((int(lm.x*roi.shape[1])+x1,
                                int(lm.y*roi.shape[0])+y1))
                self.keypoints_history.setdefault(tid, []).append(kps)

                # speed using hip (index 24)
                hip = kps[24]
                now = time.time()
                if tid in self.prev_positions:
                    px0,py0,t0 = self.prev_positions[tid]
                    dt = now - t0 or 1e-6
                    sp = math.hypot(hip[0]-px0, hip[1]-py0)/dt
                    self.speed_history.setdefault(tid, []).append(sp)
                else:
                    self.speed_history.setdefault(tid, []).append(0.0)
                self.prev_positions[tid] = (*hip, now)

                # draw landmarks and boxes
                self.draw_corner_rect(frame, (x1,y1,x2-x1,y2-y1),
                                      line_length=15, line_thickness=2,
                                      rect_thickness=1,
                                      rect_color=tuple(self.colors[0]),
                                      line_color=tuple(self.colors[0][::-1]))
                cv2.putText(frame, f"ID:{tid}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

            # draw ball
            if last_ball_bbox:
                self.draw_corner_rect(frame, tuple(last_ball_bbox),
                                      line_length=15, line_thickness=3,
                                      rect_thickness=1,
                                      rect_color=tuple(self.colors[32]),
                                      line_color=tuple(self.colors[32][::-1]))

            # write and display FPS
            self.writer.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                fps_calc = frame_count/(time.time()-start_time)
                print(f"Processed {frame_count} frames, FPS: {fps_calc:.2f}")

            # break on 'q'
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

        # Cleanup
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

        # Export CSV metrics
        self.export_metrics()


if __name__ == '__main__':
    analyzer = PlayerPerformanceAnalyzer(
        source_video_path='../data/youtube.mp4',
        output_video_path='../data/output.mp4',
        output_csv_path='../data/player_metrics.csv',
        yolov_model_path='models/yolov9s.pt'
    )
    analyzer.process_video()
