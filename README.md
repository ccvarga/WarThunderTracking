Goal:

  Create a target lead indicator to calculate and display the intercept between the player controlled aircraft's bullets and the enemy aircraft's flight vector.
  
Method: 

  Use YoloV8 to detect the pixels of the in-game target indicator to track the enemy aircraft. 
  
  Use the location of the tracked target to read the distance which is displayed below the target indicator.
  
  Use War Thunder's localhost:8111 to obatin the player aircrafts flight vector.
  
  Copy the formula for the IRL B-29 Superfortress to calculate the intercept using the obtained data.
  
  Display the intercept point for the player to shoot the weaons at.
  


Current Situation:

  The image recognition model is currently trained and image recognition works. 
  
  Development has been halted because the framerate is too low with my current setup. 
  
    Offloading image recognition model calculation to the gpu is only supported by AMD cards on linux operating system. I experemented with running two sepate applications, one in a linux virtual machine to take advantage of using the gpu for computing the image recognition, and the other application to run on my Windows OS where the video game is running. Sadly, communication between the two applications with this setup is very difficult, and it would simply be easier for me to get an Nvidia card or wait until AMD supports Windows OS.
