import argparse
import torch
from REVMark import Encoder, Decoder
import cv2

def load_video(video_path, frame_size=(128, 128), sequence_len=8):
    """
    Load a video and preprocess it into a fixed-size tensor.
    Args:
        video_path (str): Path to the video file.
        frame_size (tuple): Frame size (width, height).
        sequence_len (int): Number of frames in the sequence.
    Returns:
        torch.Tensor: Preprocessed video tensor of shape (1, 3, F, H, W).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(sequence_len):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, frame_size) / 255.0)
    cap.release()
    if len(frames) < sequence_len:
        raise ValueError(f"Video {video_path} has fewer than {sequence_len} frames.")
    return torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).float()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a video using REVMark.")
    parser.add_argument("--video_file", type=str, required=True, help="Path to the video file to be tested.")
    parser.add_argument("--encoder_path", type=str, default="checkpoints/revmark-encoder.pth", help="Path to the trained encoder model.")
    parser.add_argument("--decoder_path", type=str, default="checkpoints/revmark-decoder.pth", help="Path to the trained decoder model.")
    parser.add_argument("--sequence_len", type=int, default=8, help="Number of frames in the sequence.")
    parser.add_argument("--frame_size", type=int, nargs=2, default=(128, 128), help="Frame size (width, height).")
    args = parser.parse_args()

    # Load pretrained models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = Encoder(msgbitnum=96, videoshape=[args.sequence_len, *args.frame_size]).to(device).eval()
    decoder = Decoder(msgbitnum=96, videoshape=[args.sequence_len, *args.frame_size]).to(device).eval()
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Load and preprocess the video
    try:
        cover_video = load_video(args.video_file, frame_size=args.frame_size, sequence_len=args.sequence_len).to(device)
    except ValueError as e:
        print(e)
        exit(1)

    # Generate random watermark message
    watermark_msg = torch.randint(0, 2, (1, 96)).float().to(device)

    # Embed watermark
    residual = encoder(cover_video, watermark_msg)
    watermarked_video = (cover_video + 6.2 * residual).clamp(-1, 1)

    # Add noise for testing
    noisy_video = watermarked_video + torch.randn_like(watermarked_video) * 0.04

    # Extract watermark
    extracted_msg = decoder(noisy_video)
    accuracy = ((extracted_msg > 0.5) == (watermark_msg > 0.5)).float().mean().item()

    print(f"Tested Video: {args.video_file}")
    print(f"Extraction Accuracy: {accuracy:.2%}")
