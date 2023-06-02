from moviepy.editor import *
   
# loading video dsa gfg intro video
clip = VideoFileClip("videos/input_video.mp4")
   
# applying speed effect
final = clip.fx( vfx.speedx, 10)
  
# showing final clip
final.ipython_display()