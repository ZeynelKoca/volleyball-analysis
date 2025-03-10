# Volleyball analysis

## Dataset

### Download unstructured video from youtube

Following command downloads the highest available video quality. Requires yt-dlp and ffmpeg

```bash
yt-dlp -f "bestvideo+bestaudio/best" --merge-output-format mp4 https://www.youtube.com/URL-TO-VIDEO
```