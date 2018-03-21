Mariya Kazachkova
Computer Vision
Assignment 3


For all parts the following holds true:
batch size of 32
Adam optimizer with .0001 learning rate
For training, the data is shuffled (using the DataLoader)

Part a:

Without data augmentation: 
I ran for 15 epochs because that was how long it took the loss to stabalize (at roughly .01 - .02). 
I got a train accuracy of 100% and a test accuracy of 50%. It is very obvious that we are overfitting 
and the network is simply memorizing the images. 

With data augmentation:
I ran for 30 epochs, which was how long it took the loss to stabilize (at roughly .4 - .5).
I got a train accuracy of 98.2% and test accuracy of 53.63%. We are still very obviously overfitting, even with the data augmentation.



Part b:

Using a margin of 10, and checking to see if the distance is less than 10 when deciding if two images are of the same person (<10 = same person)

Without data augmentation:
I ran for 20 epochs until loss stabilized (roughly 5-20). 
I got a training accuracy of 89.84% and a test accuracy of 58.97%.
This is already better than random chance; our choice of loss function seems to be more appropriate here (see part c for more discussion). 

With data augmentation:
I ran for 35 epochs until the loss stabilized (roughly 20-70).
I got a training accuracy of 75.0% and a test accuracy of 64.1%.
Our test accuracy has gone up a little from the accuracy without data augmentation. This means that our network is overfitting less and learning more distingushing features rather than just memorizing the training data.  



Part c:

For part a, it seems that both with and without data augmentation I could not get a test accuracy larger than ~50%. Because data augmentation is supposed to help with generalization and thus make our test accuracy higher, it seems that the type of loss we are using might actually be the problem. Because the loss is just based on whether our prediction is right or not, it may not necessarily be learning the correct features to distinguish similarity across all images of faces, thus making our train accuracy really good but our test accuracy really bad.

In part b, we see that even withouy data augmentation our test accuracy already jumps to higher than random chance (it is ~59%). This means that even without manipulating our images to make our network more generalizable, the loss that we have chosen allows our network to place more emphasis on distinguishing features. This is because our new loss (the contrastive loss) relies on distance, meaning how "far" apart the features of an image pair are from one another. In this way the network not only uses whether the images are of the same person or not, but also how different or similar the two faces are to one another to better updates its weights. When we add data augmentation our accuracy becomes even better, moving to 64%.   



