import os
import os.path as osp
import cv2
import threading

def frame_extraction(video_path):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    # target_dir = osp.join('./tmp','spatial_skeleton_dir')
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

def thread(num_threads, id_thread):
    directory='./videos'
    files = os.listdir(directory)
    for filename in files:
        if files.index(filename) % num_threads == id_thread:
            f = os.path.join(directory, filename)
            print(f"{id_thread} - {filename}, {directory}")
            
            frame_extraction(f)

def main():
    num_threads = 8
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=thread, args=(num_threads, i))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()      
    
if __name__ == '__main__':
    main()
