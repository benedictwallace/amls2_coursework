# amls2_coursework
main.py may not run, run the seperate pre_train and main_vgg files for better results.

RUN main.py with PRETRAIN = True to pretrain the model, otherwise only the GAN phase will run
change the track with TRACK = 1 or 2.

pre_train.py is just the pretraining for track 1.
main_vgg.py is just the GAN phase for track 1.

pre_train_B.py is just the pretraining for track 2.
main_vgg_B.py is just the GAN phase for track 2.

display_model.py contains the code to stitch patches together and run them through the models, be careful with the scales when changing tracks.

The datasets need to be imported to the /datasets folder from https://data.vision.ee.ethz.ch/cvl/DIV2K/

Apologies for the bad formatting and lack of datasets, I had some issues with git last minute.
