from moviepy.editor import VideoFileClip
from IPython.display import HTML
from image_det import process_image

n = 2
if n == 1:
    test_output = 'test1.mp4'
    clip = VideoFileClip("test_video.mp4")
else:
    test_output = 'project.mp4'
    clip = VideoFileClip("project_video.mp4")

test_clip = clip.fl_image(process_image)

test_clip.write_videofile(test_output, audio=False)