# MNIST-Classifier
Look at the name of the repo, it is self-explainatory! Second ML/DL project!
- The architecture of the NN: 784 -> 10 -> 10 -> 1
- Even though the MLP was trained with 600 epochs, with 90.26% accuracy on test set. It is still, very dumb when classifying digits drawn from my mouse. This is probably because the data used to train was centered, scaled so that they looks somewhat similar. Meanwhile, drawing off by just a pixel to the left can change the network's prediction significantly.
- To produce the images below, it took me tries (more than 20) to draw 7s and 9s so that the network actually correctly recognise them, while other numbers took maximum 3 tries. (Probably because they have many overlapping features with other numbers, tricking the network into thinking of another number).
![alt text](readmeImg/training.png)
![alt text](readmeImg/0.png)
![alt text](readmeImg/1.png)
![alt text](readmeImg/2.png)
![alt text](readmeImg/3.png)
![alt text](readmeImg/4.png)
![alt text](readmeImg/5.png)
![alt text](readmeImg/6.png)
![alt text](readmeImg/7.png)
![alt text](readmeImg/8.png)
![alt text](readmeImg/9.png)