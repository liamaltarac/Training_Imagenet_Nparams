for file in ~/scratch/ILSVRC2012_img_val/*.JPEG; do 
    read line;
    d=`basename $file`
    mkdir -p ~/scratch/ILSVRC2012_img_val/${line}
    mv -v "${file}" ~/scratch/ILSVRC2012_img_val/${line}; 
 done < imagenet_utils/ILSVRC2012_validation_ground_truth_synset.txt