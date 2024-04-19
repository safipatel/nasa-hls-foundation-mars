# nasa-hls-foundation-mars
How well does the NASA/IBM HLS Geospatial Foundation Model perform on downstream applications on Mars satellite imagery?

* Check Reconstruction Results to see how first results.
* Look at config to see how our configuration for mmsegmentation looks
* convert_dataset changes from coco-segmentation format from roboflow into masks and with folders in the format that mmsegmentation needs
* Notice also: there has been some changed to Prithvi.py compared to the original one on the HLS github, this is because I needed to make some changes in order to allow for the model to evaluate if I gave them a mask
