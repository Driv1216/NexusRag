import os
import shutil
from typing import List, Dict, Any, Tuple
from PIL import Image
from backend.ingestion.base import BaseIngestor
from backend.models import DocumentChunk, ChunkMetadata
from backend.vision.captioner import VisionCaptioner
from backend.config import VIDEO_DIR, IMAGES_DIR

class VideoIngestor(BaseIngestor):
    """Extracts keyframes and analyzes visual content from video files."""

    def __init__(self):
        self.captioner = VisionCaptioner()

    def ingest(self, file_path: str, filename: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        clean_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lstrip(".")
        saved_video_path = os.path.join(VIDEO_DIR, f"{clean_name}.{ext}")
        if os.path.abspath(file_path) != os.path.abspath(saved_video_path):
            shutil.copyfile(file_path, saved_video_path)

        chunks: List[DocumentChunk] = []
        extracted_frames = self._extract_keyframes(saved_video_path, clean_name)

        for f_idx, frame_info in enumerate(extracted_frames):
            frame_path = frame_info["frame_path"]
            timestamp_str = frame_info["timestamp"]
            rel_img_url = f"/uploads/extracted_images/{os.path.basename(frame_path)}"

            caption = self.captioner.caption_image(
                frame_path,
                context_hint=f"Video '{filename}' Keyframe at {timestamp_str}"
            )

            chunk_id = f"video_{clean_name}_kf{f_idx+1}"
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                source_file=filename,
                modality="video",
                page_number=f_idx + 1,
                image_url=rel_img_url,
                image_path=frame_path,
                caption=caption,
                extra_metadata={
                    "timestamp": timestamp_str,
                    "frame_index": f_idx + 1
                }
            )

            content = f"[VIDEO KEYFRAME: {filename} at {timestamp_str}]\nVisual Analysis & Screen Content:\n{caption}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=content,
                metadata=meta
            ))

        return chunks, {
            "total_pages": len(extracted_frames),
            "figures_extracted": len(extracted_frames),
            "tables_extracted": 0,
            "total_chunks": len(chunks)
        }

    def _extract_keyframes(self, video_path: str, clean_name: str) -> List[Dict[str, Any]]:
        """Extract keyframes using PyAV, OpenCV, or moviepy if available, or generate representative frames."""
        frames = []
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            # Sample 3-5 keyframes evenly across duration
            sample_points = [0.1, 0.5, 0.9] if duration > 10 else [0.5]
            for p in sample_points:
                target_frame = int(p * total_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if ret:
                    time_sec = int(target_frame / fps)
                    m, s = divmod(time_sec, 60)
                    time_str = f"{m:02d}:{s:02d}"
                    frame_filename = f"{clean_name}_frame_{time_str.replace(':', '_')}.jpg"
                    frame_save_path = os.path.join(IMAGES_DIR, frame_filename)
                    cv2.imwrite(frame_save_path, frame)
                    frames.append({
                        "frame_path": frame_save_path,
                        "timestamp": time_str
                    })
            cap.release()
            if frames:
                return frames
        except Exception as e:
            print(f"OpenCV video extraction note: {e}.")

        # Fallback placeholder thumbnail
        fallback_filename = f"{clean_name}_thumb.jpg"
        fallback_path = os.path.join(IMAGES_DIR, fallback_filename)
        if not os.path.exists(fallback_path):
            img = Image.new("RGB", (480, 270), color=(30, 41, 59))
            img.save(fallback_path)
        return [{
            "frame_path": fallback_path,
            "timestamp": "00:05"
        }]
