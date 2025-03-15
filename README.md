# Volleyball analysis

## Dataset

### Download unstructured video from youtube

Following command downloads the highest available video quality. Requires yt-dlp and ffmpeg

```bash
yt-dlp -f "bestvideo+bestaudio/best" --merge-output-format mp4 https://www.youtube.com/URL-TO-VIDEO
```

### Split unstructured video into separate clips

Currently use Davinci Resolve. This is not the most efficient because all clips have to be re-coded (re-rendered) because no project currently exists that easily lets us split a video and output into separate clips without having to re-code

1. Split full video into separate clips for label `serve`, `play`, `idle`
2. Export as separate clips
3. Place clips in dataset directory into their respective label folders `train`, `val`, `test`

```md
gamestate-dataset/
├── train/
│   ├── play
│   ├── serve
│   └── idle
├── val/
│   ├── play
│   ├── serve
│   └── idle
└── test/
    ├── play
    ├── serve
    └── idle
```