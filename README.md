# <u>Face Recognition System
Face Recognition system using **Siamese Neural network**. The model is based on the **FaceNet model**. 
**OpenCV** implementaion has been done for realtime face detection and recognition using the network. The model uses face encodings for identifying users.
  
 There are two options:
 1. **Realtime facial recognition:** In this the model does facial recognition in realtime using the camera feed with text overlay telling whether the user is registered with the system or not.
 2. **Normal facial recognition:** In this the camera starts for 5-6 seconds and snaps some pictures of the user, then the picture is used for recognition.
 
  
  
#### References:
- Code for Facenet model is based on the assignment from Convolutional Neural Networks Specialization by Deeplearning.ai on Coursera.<br>
https://www.coursera.org/learn/convolutional-neural-networks/home/welcome 
- Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
- Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
- The pretrained model used is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
- A lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
