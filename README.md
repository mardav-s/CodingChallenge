# CodingChallenge
![answer](https://github.com/user-attachments/assets/ff7c7b70-e23f-45a7-ba25-4e8719d30add)

Perception Coding Challenge
The methodology involves detecting cone-like shapes in an image by applying color thresholding, edge detection, and contour analysis. After cleaning the image and simplifying contours, convex hulls are filtered and identified as cones based on their orientation. Finally, lines are fitted to the detected cones using least squares, and the results are visualized.
I had a lot of issues using the libraries correctly. I was rusty in Python and unfamiliar with some of the libraries I was suggested to use. I did a lot of research on the specific methods that I could use from each library. For example, some of the numpy methods I used such as zeros_like, I had to do further research on. Many of the CV methods I used required the same. 
Libraries used: OpenCV, NumPy, SciPy, MatPlotLib
