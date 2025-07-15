import os
import os.path as osp
import subprocess

import mmengine
data_root = '/home/peco/data/gym/'
video_root = f'{data_root}/videos'
videos = os.listdir(video_root)
videos = set(videos)
for video in videos:
    if not video.endswith(".mp4"):
        
        video_path=os.path.join(video_root,video)
        output_video_name=video.split('.')[0]+'.mp4'
        output_video_path=os.path.join(video_root,output_video_name)
        command = [
        'ffmpeg', '-i',
        '"%s"' % video_path, '-c','copy',
        '"%s"' % output_video_path
        ]
        command = ' '.join(command)
        try:
            subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(
                f'Error',
                flush=True)