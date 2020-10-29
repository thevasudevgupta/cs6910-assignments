# Q1

Please train a network for image classification on your respective datasets (please ref Sec. 3). You should experiment with the following network parameters:
1. number of convolutional (conv) layers 2. fully connected (fc) layers
3. number of filters in different layers
4. maxpooling
5. training time (number of epochs)
6. stride
to come up a study of the effect of these parameters on the classification performance. Try to improve the performance as much as possible by modifying these parameters. Please present the results of such a study in the form of a table that shows the classification performance as a function of these parameters. Also look at some of the images that are mis-classified and see if there is an explanation for such mis-classifications.

# Directory structure

    .
    ├── config.py           # Configuration used in model defining and training 
    ├── main.py             # run this file for starting training
    ├── dataloader.py       # all dataset related stuff
    ├── trainer.py          # all training related stuff

**Submitted by: Vasudev Gupta (ME18B182)**